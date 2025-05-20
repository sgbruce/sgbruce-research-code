# Implementation of CCE Estimation in a Bid-Based Trading Environment (TensorFlow)
import numpy as np
import tensorflow as tf
import cvxpy as cp
import itertools
import time
import json
import os
from datetime import datetime
import dubey_regret as dubey

# --- Configure NumPy string formatting ---
# This prevents NumPy from truncating large arrays with '...'
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

'''
    This dictionary contains the parameters for the JPSRO algorithm. Currently 
    it operates the algorithm for max_epochs iterations, pruning the strategy set
    every pruning_parameter epochs. For the BR operator, it samples
    best_response_samples joint strategies from the meta-game and optimizes the
    BR for that many joint strategies. It then runs the BR for best_response_epochs
    epochs, using the optimizer specified by solver and learning_rate.
'''
PARAM_DICT = {
    "solver_name": "RMSProp",
    "solver": tf.keras.optimizers.RMSprop,
    "amsgrad": True,
    "learning_rate": 0.001,
    "max_epochs": 25,
    "best_response_epochs": 200,
    "best_response_samples": 500,
    "pruning_parameter": 4
}

'''
    This class is a custom trading environment that inherits from the DubeyGame class.
    It is used to compute the trading amounts for a set of bids, and the utilities a player 
    recieves given said bids, endowments, and cobb-douglas exponents.

    It is specificed for a single market state E, and is initialized with a list of endowments of size 
    (num_players, num_goods), and a list of cobb-douglas exponents of size (num_players, num_goods), 
    such that for exponents x,y and goods a,b, a player's utility is given a^x * b^y - beta, 
    where beta is the net credit after trade if negative. 
'''
class BidTradingEnv(dubey.DubeyGame):
    '''
    Initialize the market environment E with a list of endowments and a list of cobb-douglas exponents.
    '''
    def __init__(self, endowments, cobb_douglas_exponents):
        super().__init__(len(endowments[0]))
        self.num_goods = len(endowments[0])
        self.endowments = endowments  # list of shape (num_players, num_goods)
        self.num_players = len(endowments)
        self.alphas = tf.convert_to_tensor(cobb_douglas_exponents, dtype=tf.float32)  # list of shape (num_players, num_goods)
        self.reset()

    '''
        Reset the market environment E to the initial state, for now is just a dummy.
        If you want to implement a dynamic market state over time, you can do so here.
    '''
    def reset(self):
        return np.zeros((len(self.endowments), self.num_goods))  # placeholder obs
    
    '''
        Given a list of bids, compute the trading amounts, and the utilities a player recieves.
        Returns the tuple (next state, rewards, done, info).

        next state here is none, since the environment is static. If you want to implement a dynamic market state,
        you can do so here.

        rewards is a tensor of shape (num_players,) containing the utility each player recieves after the trade.

        done is a boolean indicating if the episode is over (always true in static case).

        info is a dictionary containing additional information about the trade. Here we include 
        the actual executed bids (quantities and prices traded), the allocations of goods to players,
        a boolean tensor of shape (num_players,) indicating if the bid was valid, and the net credit
        of each player.
    '''
    def step(self, bids: tf.Tensor, debug = False):
        # bids = list of (buy_price, buy_qty, sell_price, sell_qty) * num_goods per agent
        executed_bids = self.compute_trade_amounts(bids) # compute the executed bids using the DubeyGame class
        endowments = tf.cast(self.endowments, dtype=tf.float32)
        # build allocations as purchase amount + endowment - sell amount
        bought_prices = tf.squeeze(executed_bids[:, 0, :]) # Shape: (num_players, num_goods)
        bought_quantities = tf.squeeze(executed_bids[:, 1, :]) # Shape: (num_players, num_goods)
        sold_prices = tf.squeeze(executed_bids[:, 2, :]) # Shape: (num_players, num_goods)
        sold_quantities = tf.squeeze(executed_bids[:, 3, :])   # Shape: (num_players, num_goods)

        # Calculate allocations element-wise using TensorFlow
        allocations = bought_quantities + endowments - sold_quantities # Shape: (num_players, num_goods)
        net_credit = tf.reduce_sum(sold_prices * sold_quantities - bought_prices * bought_quantities, axis=1) # Shape: (num_players,)

        rewards = self.compute_utilities(allocations, net_credit)

        # check if bids are valid (non-negative and selling no more than owned)
        sell_quantities = tf.squeeze(bids[:, 3, :])
        valid_bids = tf.logical_and(tf.reduce_all(bids >= 0, axis=(1, 2)), tf.reduce_all(endowments - sell_quantities >= 0, axis=1))
        valid_bids = tf.cast(valid_bids, tf.bool)

        return None, rewards, True, {"executed_bids": executed_bids, "allocations": allocations, "valid_bids": valid_bids, "net_credit": net_credit}
    

    '''
        Vector step does the same computation as step, but for a batch of bids. The outputs are as above,
        however each returned tensor has an added dimension at axis 0, corresponding to the batch size.

        Processing the bids in batches allows for parallel processing, and is more efficient for many sets of bids.
    '''
    def vector_step(self, bids, debug = False):
        # bids = list of (buy_price, buy_qty, sell_price, sell_qty) * num_goods per agent
        executed_bids = self.compute_trade_amounts_vectorized(bids) # compute the trade amounts in batch form, using the DubeyGame class
        if debug:
            executed_bids = tf.debugging.check_numerics(executed_bids, "NaN/Inf in executed_bids") # Add check
        
        endowments = tf.cast(self.endowments, dtype=tf.float32)
        # build allocations as purchase amount + endowment - sell amount
        bought_prices = tf.squeeze(executed_bids[:, :, 0, :]) # Shape: (batch_size, num_players, num_goods)
        bought_quantities = tf.squeeze(executed_bids[:, :, 1, :]) # Shape: (batch_size, num_players, num_goods)
        sold_prices = tf.squeeze(executed_bids[:, :, 2, :]) # Shape: (batch_size, num_players, num_goods)
        sold_quantities = tf.squeeze(executed_bids[:, :, 3, :])   # Shape: (batch_size, num_players, num_goods)

        # Calculate allocations element-wise using TensorFlow
        allocations = bought_quantities + endowments - sold_quantities # Shape: (batch_size, num_players, num_goods)
        net_credit = tf.reduce_sum(sold_prices * sold_quantities - bought_prices * bought_quantities, axis=2) # Shape: (batch_size, num_players)

        rewards = self.compute_utilities_vectorized(allocations, net_credit) # Shape: (batch_size, num_players)
        if debug:
            rewards = tf.debugging.check_numerics(rewards, "NaN/Inf in rewards") # Add check

        # check if bids are valid (non-negative and selling no more than owned)
        sell_quantities = tf.squeeze(bids[:, :, 3, :])
        valid_bids = tf.logical_and(tf.reduce_all(bids >= 0, axis=(2, 3)), tf.reduce_all(endowments - sell_quantities >= 0, axis=2))
        valid_bids = tf.cast(valid_bids, tf.bool)

        return None, rewards, True, {"executed_bids": executed_bids, "allocations": allocations, "valid_bids": valid_bids, "net_credit": net_credit}

    '''
        Compute the utilities for a given set of allocations and net credit. Uses the cobb-douglass 
        exponents to compute the base utility for each player.
    '''
    def compute_utilities(self, allocations, net_credit):
        # allocations is a tensor of shape (num_players, num_goods)
        # net_credit is a tensor of shape (num_players,)
        utilities = tf.reduce_prod(allocations ** self.alphas, axis=1) + 0.5 * tf.minimum(0, net_credit)
        return utilities
    
    '''
        Vectorized version of compute_utilities. Computes the utilities for a batch of allocations and net credit.
    '''
    def compute_utilities_vectorized(self, allocations, net_credit):
        # allocations is a tensor of shape (batch_size, num_players, num_goods)
        # net_credit is a tensor of shape (batch_size, num_players)
        epsilon = 1e-9 # Small epsilon to avoid log(0) or pow(<0, frac) issues
        safe_allocations = tf.maximum(allocations, epsilon)
        expanded_alphas = tf.expand_dims(self.alphas, axis=0)
        tiled_alphas = tf.tile(expanded_alphas, multiples=[tf.shape(allocations)[0], 1, 1])
        utilities = tf.reduce_prod(safe_allocations ** tiled_alphas, axis=2) + 0.5 * tf.minimum(0, net_credit)
        return utilities

    '''
        Evaluate the utilities for a given joint strategy tuple and returns net utilities for each player.
    '''
    def evaluate_policy_tuple(self, policies):
        _, rewards, _, _ = self.step(tf.convert_to_tensor(policies, dtype=tf.float32))
        return rewards

    '''
        Create a basic policy for a given player index. This is a random policy that is used to initialize the policy set.
    '''    
    def create_basic_policy(self, player_idx):
        base_strat = [np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 1.5, np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 1.5]
        return tf.convert_to_tensor(base_strat, dtype=tf.float32)
    

'''
    Train the best response for a given player index. This is done by sampling a set of joint strategies from the joint distribution,
    and then optimizing a single player's bid for the given player index.

    optimizing is done using a simple gradient descent optimizer. Parameters for the optimizer are specified in PARAM_DICT.
'''
def train_best_response(env: BidTradingEnv, joint_distribution, player_idx, endowment, debug = False):
    # sample a strategy with p according to the joint distribution
    def sample_joint_strategy(joint_distribution):
        keys = [joint_distribution[i][0] for i in range(len(joint_distribution))]
        probabilities = [joint_distribution[i][1] for i in range(len(joint_distribution))]
        selected_key = np.random.choice([i for i in range(len(joint_distribution))], p=probabilities)
        return keys[selected_key]
    
    # Step 1: Sample opponent bid profiles
    opponent_samples = []
    for _ in range(PARAM_DICT["best_response_samples"]):
        joint = sample_joint_strategy(joint_distribution)  # list of agent policies
        opp_bids = tf.convert_to_tensor([pi for i, pi in enumerate(joint) if i != player_idx], dtype=tf.float32)
        opponent_samples.append(opp_bids)
    opponent_samples = tf.convert_to_tensor(opponent_samples, dtype=tf.float32)
    
    if debug:
        print("opponent_samples generated: ", len(opponent_samples))
    
    # Step 2: Intialize agent's bid
    agent_bid = tf.Variable(tf.random.uniform([4, 2], maxval=[[1, 1], endowment, [1, 1], endowment]), dtype=tf.float32)

    # Step 3: Optimize agent's bid
    if PARAM_DICT["solver_name"] == "Adam":
        optimizer = PARAM_DICT["solver"](learning_rate=PARAM_DICT["learning_rate"], amsgrad=PARAM_DICT["amsgrad"])
    else:
        optimizer = PARAM_DICT["solver"](learning_rate=PARAM_DICT["learning_rate"])
    
    for epoch in range(PARAM_DICT["best_response_epochs"]):  # optimization steps
        with tf.GradientTape() as tape:
            total = 0
            # turn bids into batches so all joint strategies are processed in parallel
            agent_bid_expanded = tf.expand_dims(tf.expand_dims(agent_bid, axis=0), axis=0)
            batched_bids = tf.tile(agent_bid_expanded, multiples=[opponent_samples.shape[0], 1, 1, 1])
            all_bids = tf.concat([opponent_samples[:, :player_idx], batched_bids, opponent_samples[:, player_idx:]], axis=1)
            # compute the utilities of the bids
            _, utils, _, info = env.vector_step(all_bids, debug = debug) # takes ~0.015s
            total = tf.reduce_sum(utils[:, player_idx])
            # loss is the negative of the average utility of the player
            loss = -total / tf.cast(opponent_samples.shape[0], tf.float32) # Cast divisor just in case
            if debug:
                loss = tf.debugging.check_numerics(loss, "NaN/Inf in loss calculation") # Add check
        grads = tape.gradient(loss, [agent_bid])
        if debug:
            grads = [tf.debugging.check_numerics(g, f"NaN/Inf in gradient for {agent_bid.name}") if g is not None else g for g in grads] # Add check
        optimizer.apply_gradients(zip(grads, [agent_bid]))
        if debug and epoch % 10 == 0:
            print("\n\nepoch: ", epoch, "loss: ", loss)
            print("agent_bid: \n", agent_bid.numpy())
            print("executed_bids: \n", info["executed_bids"][0].numpy())
            print("allocations: \n", info["allocations"][0].numpy())
    
    return agent_bid

'''
    Build the meta-game for a given sets of policies. This is done by taking the Cartesian product of the policies,
    and then evaluating the utilities of each joint policy.
'''
def build_meta_game(env: BidTradingEnv, policy_sets):
    joint_policies = list(itertools.product(*policy_sets))
    meta_game = []
    for i, joint in enumerate(joint_policies):
        payoff = env.evaluate_policy_tuple(joint)
        meta_game.append([joint, payoff])
    return meta_game

'''
    Solve the CCE for a given meta-game. This is done using linear programming. To build the constraints, add one 
    constraint for each player for each fixed deviation. The values of each constraint correspond to the utility of the 
    deviation. This deviation utility times the probabilities should be less than 0.

    The resulting probability distribution can be maximized for welfare or gini. 
'''
def solve_cce(meta_game, num_players, policy_sets):
    joint_strats = [meta_game[i][0] for i in range(len(meta_game))] # List of tuples of Tensors
    # Ensure payoffs are numpy for indexing and calculations within the solver
    joint_payoffs = np.array([meta_game[i][1].numpy() for i in range(len(meta_game))])
    num_strats = len(joint_strats)

    # Helper function to compare two strategies (tensors) for equality
    def compare_strategies(strat1, strat2):
        s1 = tf.convert_to_tensor(strat1)
        s2 = tf.convert_to_tensor(strat2)
        return tf.reduce_all(tf.equal(s1, s2)).numpy()

    # Helper function to compare two joint strategies (tuples of tensors)
    def compare_joint_strategies(joint_strat1, joint_strat2):
        if len(joint_strat1) != len(joint_strat2):
            return False
        for p in range(len(joint_strat1)):
            if not compare_strategies(joint_strat1[p], joint_strat2[p]):
                return False
        return True

    sigma = cp.Variable(num_strats)
    # Objective: Maximize entropy (or minimize negative entropy proxy) for a less extreme CCE
    # Using a simple quadratic form (-0.5 * ||sigma||^2) encourages smoother distributions
    objective = cp.Maximize(cp.sum(joint_payoffs.T @ sigma)) # MWCCE
    # objective = cp.Maximize(-0.5 * cp.sum_squares(sigma)) # MGCCE
    constraints = [cp.sum(sigma) == 1, sigma >= 0]

    for p in range(num_players):
        for alt_pi in policy_sets[p]: # alt_pi is a Tensor representing a strategy
            constraint_sum = 0
            for i, pi in enumerate(joint_strats): # pi is a tuple of Tensors representing a joint strategy
                original_payoff = joint_payoffs[i, p] 

                # Check if the alternative strategy is the same as the current one for player p
                if compare_strategies(alt_pi, pi[p]):
                    continue
                
                # Construct the alternative joint strategy tuple
                pi_alt_list = list(pi)
                pi_alt_list[p] = alt_pi
                pi_alt_tuple = tuple(pi_alt_list)

                # Find the index of this alternative strategy in joint_strats
                found_alt = False
                alt_index = -1
                for idx, js in enumerate(joint_strats):
                    if compare_joint_strategies(pi_alt_tuple, js):
                        found_alt = True
                        alt_index = idx
                        break

                if not found_alt:
                    # This deviation is not possible within the pre-calculated meta_game
                    print(f"Warning: Alt strategy not found in meta_game for constraint check.") 
                    continue

                else:
                    # Get the payoff for the alternate strategy using the found index
                    alt_payoff = joint_payoffs[alt_index, p]

                # Accumulate the expected gain/loss from deviation for this joint strategy i
                constraint_sum += sigma[i] * (alt_payoff - original_payoff)

            # Add the CCE constraint: Expected payoff from deviating should not be greater than original
            constraints.append(constraint_sum <= 1e-4) # Use a small tolerance

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve()
    except Exception as e:
        print(f"Error: {e}")
        return [(joint_strats[i], 1.0/num_strats) for i in range(num_strats)]

    # Check solver status
    if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Warning: CCE Solver failed or found inaccurate solution. Status: {prob.status}")
        return [(joint_strats[i], 1.0/num_strats) for i in range(num_strats)]

    # Handle case where sigma.value might be None if solver fails
    sigma_value = sigma.value if sigma.value is not None else np.zeros(num_strats)
    # Normalize sigma_value just in case of small numerical errors
    sigma_value = np.maximum(sigma_value, 0) # Ensure non-negative
    sigma_sum = np.sum(sigma_value)
    if sigma_sum > 1e-6:
        sigma_value /= sigma_sum
    else: # Handle case of all zeros (e.g., solver failure)
        sigma_value = np.ones(num_strats) / num_strats

    return [(joint_strats[i], sigma_value[i]) for i in range(len(joint_strats))]

'''
    Prune the strategies that are not played with positive probability. This is done by counting the probability of each strategy
    in the joint distribution. If a strategy is played with zero probability, it is removed from the policy set.
'''
def prune_strategies(policy_sets, sigma):
    # Count the probability of each strategy in the joint distribution, shape (num_players, num_strategies)
    policy_counts = [[0 for _ in range(len(policy_sets[p]))] for p in range(len(policy_sets))]
    for joint_strat, prob in sigma:
        for i, pi in enumerate(joint_strat):
            for j, strat in enumerate(policy_sets[i]):
                if np.all(pi == strat):
                    policy_counts[i][j] += prob
    new_policy_sets = []
    # Now for each player recreate the policy set with only the strategies that are played with positive probability
    for p in range(len(policy_sets)):
        new_player_policies = []
        for i, pi in enumerate(policy_sets[p]):
            if policy_counts[p][i] > 1e-6:
                new_player_policies.append(pi)
            else: 
                print(f"Player {p} policy {i} has zero probability. Removing.")
        new_policy_sets.append(new_player_policies)
    return new_policy_sets

'''
    Run the JPSRO algorithm. This is done by building the meta-game, solving the CCE, and then using the best response to add a new policy
    to the policy set. This is done iteratively until the policy sets converge.

    Every pruning_parameter steps, the policy sets are pruned to remove strategies that are not played.
'''
def jprso(env: BidTradingEnv, debug = False, log_file = None):
    # Initialize the algorithm and create step 0 meta-game and CCE
    num_players = env.num_players
    policy_sets = [[env.create_basic_policy(p)] for p in range(num_players)]
    meta_game = build_meta_game(env, policy_sets)
    sigma = solve_cce(meta_game, num_players, policy_sets)

    if debug:
        print("initial Configuration:")
        print("policy_sets: ", [[pi.numpy() for pi in policy_sets[p]] for p in range(num_players)])
        print("meta_game: ", [[i, payoff] for i, [joint, payoff] in enumerate(meta_game)])
        print("sigma: ", [[i, sigma[i][1]] for i in range(len(sigma))])
    elif log_file:
        with open(log_file, "a") as f:
            f.write("Initial Configuration:\n")
            policy_sets_str = np.array2string(np.array(policy_sets))
            f.write(f"policy_sets: {policy_sets_str}\n")
            meta_game_str = np.array2string(tf.convert_to_tensor([payoff for i, [joint, payoff] in enumerate(meta_game)]).numpy())
            f.write(f"meta_game: {meta_game_str}\n")
            sigma_str = np.array2string(tf.convert_to_tensor([sigma[i][1] for i in range(len(sigma))]).numpy())
            f.write(f"sigma: {sigma_str}\n")
            f.write("--------------------------------\n")

    # iterate and find new best response for each player, then solve the new CCE, then prune the strategies if necessary
    for epoch in range(PARAM_DICT["max_epochs"]):
        t = time.time()
        # for each player, find a new best response policy to the previous joint distribution
        for p in range(num_players):
            # train_best_response returns a tf.Variable, convert to numpy for logging
            new_pi_var = train_best_response(env, sigma, p, env.endowments[p], debug=debug)
            new_pi_np = new_pi_var.numpy() # Get numpy array from the Variable/Tensor

            if debug:
                print(f"new policy for player {p}:\n{new_pi_np}")
            elif log_file:
                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch}, Player {p} new policy:\n{str(new_pi_np)}\n")
            # Append the original tf.Variable/Tensor to policy_sets
            policy_sets[p].append(new_pi_var)

        # build the new meta-game and solve the new CCE
        meta_game = build_meta_game(env, policy_sets)
        sigma = solve_cce(meta_game, num_players, policy_sets)

        # prune the strategies if necessary
        if epoch % PARAM_DICT["pruning_parameter"] == 0:
            policy_sets = prune_strategies(policy_sets, sigma)

        if not debug and log_file:
            with open(log_file, "a") as f:
                f.write("--------------------------------\n")
                f.write(f"End of epoch {epoch}:\n")
                policy_sets_str = np.array2string(np.array(policy_sets))
                f.write(f"policy_sets: {policy_sets_str}\n")
                meta_game_str = np.array2string(tf.convert_to_tensor([payoff for i, [joint, payoff] in enumerate(meta_game)]).numpy())
                f.write(f"meta_game: {meta_game_str}\n")
                sigma_str = np.array2string(tf.convert_to_tensor([sigma[i][1] for i in range(len(sigma))]).numpy())
                f.write(f"sigma: {sigma_str}\n")
                f.write("--------------------------------\n")
        print(f"Epoch {epoch} took {round(time.time() - t, 2)} seconds")
    return sigma

'''
    Testing function to run the JPSRO algorithm. Runs a single training run with the defined parameters.
    Returns the final joint distribution.
'''
def run_training(debug = False, log_file = None):
    # === Main Loop ===
    endowments = np.array([[4, 4], [4, 4], [4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75], [0.66, 0.33], [0.33, 0.66]])
    # optimal outcome is p=[1,1], x=[[6,2],[2,6],[5.33,2.67],[2.67,5.33]]
    env = BidTradingEnv(endowments, alphas)

    if log_file:
        with open(log_file, "a") as f:
            f.write(f"Beginning Log for {log_file}. at time {datetime.now().strftime('%Y%m%d_%H%M%S')}. Training run.\n")
            f.write("ENVIRONMENT:\n")
            f.write(f"endowments:\n{np.array2string(endowments)}\n")
            f.write(f"alphas:\n{np.array2string(alphas)}\n")
            f.write("--------------------------------\n")

    ret = jprso(env, debug = debug, log_file = log_file)
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"End of training at time: {datetime.now().strftime('%Y%m%d_%H%M%S')}\n")
    return ret

'''
    Run a parameter sweep over the hyperparameters. This is the sweep done in the thesis paper,
    results can be found there. 
'''
def run_parameter_sweep(debug = False, log_file = None):
    optimizers = [[tf.keras.optimizers.RMSprop, "RMSprop"]]#, [tf.keras.optimizers.Adam, "Adam"], [tf.keras.optimizers.Adam, "Adam"], [tf.keras.optimizers.SGD, "SGD"]]
    learning_rates = [0.005]#[0.001, 0.005]

    # Vary these after finding the best optimizer
    max_epochs = [30, 50] #[10, 20, 30, 50]
    best_response_epochs = [50, 100, 200, 500]
    best_response_samples = [100, 200, 500, 1000]

    log_prefix = f"jpsro_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(f"logs/{log_prefix}", exist_ok=True)
    os.makedirs(f"results/{log_prefix}", exist_ok=True)

    adam_seen = False
    for optimizer, optimizer_name in optimizers:
        if optimizer_name == "Adam" and adam_seen:
            PARAM_DICT["amsgrad"] = True
        elif optimizer_name == "Adam":
            PARAM_DICT["amsgrad"] = False
            adam_seen = True
        for learning_rate in learning_rates:
            PARAM_DICT["solver"] = optimizer
            PARAM_DICT["solver_name"] = optimizer_name
            PARAM_DICT["learning_rate"] = learning_rate
            for max_epoch in max_epochs:
                PARAM_DICT["max_epochs"] = max_epoch
                for best_response_epoch in best_response_epochs:
                    PARAM_DICT["best_response_epochs"] = best_response_epoch
                    for best_response_sample in best_response_samples:
                        PARAM_DICT["best_response_samples"] = best_response_sample
                        if max_epoch == 30 and best_response_epoch < 500:
                            continue
                        
                        optimizer_name = PARAM_DICT["solver_name"]
                        learning_rate = PARAM_DICT["learning_rate"]
                        # Add parentheses around the entire conditional expression
                        log_filename = f"jprso_{(optimizer_name + '_amsgrad' if PARAM_DICT['amsgrad'] and optimizer_name == 'Adam' else optimizer_name)}_{learning_rate}_{PARAM_DICT['max_epochs']}_{PARAM_DICT['best_response_epochs']}_{PARAM_DICT['best_response_samples']}"
                        sigma = run_training(debug = False, log_file = f"logs/{log_prefix}{log_filename}.txt")
                        sigma_processed = [[[strat.numpy().tolist() for strat in joint], prob] for joint, prob in sigma]
                        with open(f"results/{log_prefix}{log_filename}.json", "a") as f:
                            f.write(f"{json.dumps(sigma_processed)}\n")
    

if __name__ == "__main__":
    #run_training(debug = False, log_file = None)
    log_prefix = f"jpsro_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(f"logs/{log_prefix}", exist_ok=True)
    os.makedirs(f"results/{log_prefix}", exist_ok=True)
    log_filename = "jprso_many_player"
    sigma = run_training(debug = False, log_file = f"logs/{log_prefix}{log_filename}.txt")
    sigma_processed = [[[strat.numpy().tolist() for strat in joint], prob] for joint, prob in sigma]
    with open(f"results/{log_prefix}{log_filename}.json", "a") as f:
        f.write(f"{json.dumps(sigma_processed)}\n")