# Implementation of CCE Estimation in a Bid-Based Trading Environment (TensorFlow)
import numpy as np
import tensorflow as tf
import cvxpy as cp
import itertools
import time
from datetime import datetime
import dubey_regret as dubey

# --- Configure NumPy string formatting ---
# This prevents NumPy from truncating large arrays with '...'
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

# === Custom Trading Environment ===
class BidTradingEnv(dubey.DubeyGame):
    def __init__(self, endowments, cobb_douglas_exponents):
        super().__init__(2)
        self.num_goods = 2
        self.endowments = endowments  # list of shape (num_players, num_goods)
        self.num_players = len(endowments)
        self.alphas = tf.convert_to_tensor(cobb_douglas_exponents, dtype=tf.float32)  # list of shape (num_players, num_goods)
        self.reset()

    def reset(self):
        return np.zeros((len(self.endowments), self.num_goods))  # placeholder obs

    def step(self, bids: tf.Tensor, debug = False):
        # bids = list of (buy_price, buy_qty, sell_price, sell_qty) * num_goods per agent
        executed_bids = self.compute_trade_amounts(bids)
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

        sell_quantities = tf.squeeze(bids[:, 3, :])
        valid_bids = tf.logical_and(tf.reduce_all(bids >= 0, axis=(1, 2)), tf.reduce_all(endowments - sell_quantities >= 0, axis=1))
        valid_bids = tf.cast(valid_bids, tf.bool)

        return None, rewards, True, {"executed_bids": executed_bids, "allocations": allocations, "valid_bids": valid_bids, "net_credit": net_credit}
    
    def vector_step(self, bids, debug = False):
        # bids = list of (buy_price, buy_qty, sell_price, sell_qty) * num_goods per agent
        executed_bids = self.compute_trade_amounts_vectorized(bids)
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

        sell_quantities = tf.squeeze(bids[:, :, 3, :])
        valid_bids = tf.logical_and(tf.reduce_all(bids >= 0, axis=(2, 3)), tf.reduce_all(endowments - sell_quantities >= 0, axis=2))
        valid_bids = tf.cast(valid_bids, tf.bool)

        return None, rewards, True, {"executed_bids": executed_bids, "allocations": allocations, "valid_bids": valid_bids, "net_credit": net_credit}

    def compute_utilities(self, allocations, net_credit):
        # allocations is a tensor of shape (num_players, num_goods)
        # net_credit is a tensor of shape (num_players,)
        utilities = tf.reduce_sum(allocations ** self.alphas, axis=1) + 0.5 * tf.minimum(0, net_credit)
        return utilities
    
    def compute_utilities_vectorized(self, allocations, net_credit):
        # allocations is a tensor of shape (batch_size, num_players, num_goods)
        # net_credit is a tensor of shape (batch_size, num_players)
        epsilon = 1e-9 # Small epsilon to avoid log(0) or pow(<0, frac) issues
        safe_allocations = tf.maximum(allocations, epsilon)
        expanded_alphas = tf.expand_dims(self.alphas, axis=0)
        tiled_alphas = tf.tile(expanded_alphas, multiples=[tf.shape(allocations)[0], 1, 1])
        utilities = tf.reduce_sum(safe_allocations ** tiled_alphas, axis=2) + 0.5 * tf.minimum(0, net_credit)
        return utilities

    def evaluate_policy_tuple(self, policies):
        _, rewards, _, _ = self.step(tf.convert_to_tensor(policies, dtype=tf.float32))
        return rewards
    
    def create_basic_policy(self, player_idx):
        base_strat = [np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 2, np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 2]
        return tf.convert_to_tensor(base_strat, dtype=tf.float32)
    
# === Policy Representation === maybe change this to a more concrete model
class NNPolicy():
    def __init__(self, obs_dim, act_dim):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(act_dim)
        ])
        self.trainable_variables = self.model.trainable_variables

    def __call__(self, obs):
        return self.model(obs)

def create_policy_model_nn(obs_dim, act_dim):
    return NNPolicy(obs_dim, act_dim)



# === Train Best Response ===
def train_best_response_nn(env: BidTradingEnv, policy_set_opponents, joint_distribution,player_idx, obs_dim, act_dim):
    br_policy = create_policy_model_nn(obs_dim, act_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(50):
        obs = env.reset()
        done = False
        while not done:
            with tf.GradientTape() as tape:
                logits = br_policy(tf.convert_to_tensor([obs[player_idx]], dtype=tf.float32))
                action = tf.squeeze(logits).numpy()
                actions = []
                for i, pi in enumerate(policy_set_opponents):
                    actions.append(tf.squeeze(pi(obs[i:i+1])).numpy() if i != player_idx else action)
                _, rewards, done, _ = env.step(actions)
                loss = -rewards[player_idx]  # maximize own utility
            grads = tape.gradient(loss, br_policy.trainable_variables)
            optimizer.apply_gradients(zip(grads, br_policy.trainable_variables))
    return br_policy

def train_best_response(env: BidTradingEnv, joint_distribution, player_idx, endowment, debug = False):
    def sample_joint_strategy(joint_distribution):
        keys = [joint_distribution[i][0] for i in range(len(joint_distribution))]
        probabilities = [joint_distribution[i][1] for i in range(len(joint_distribution))]
        selected_key = np.random.choice([i for i in range(len(joint_distribution))], p=probabilities)
        return keys[selected_key]
    
    # Step 1: Sample opponent bid profiles
    opponent_samples = []
    for _ in range(200):
        joint = sample_joint_strategy(joint_distribution)  # list of agent policies
        opp_bids = tf.convert_to_tensor([pi for i, pi in enumerate(joint) if i != player_idx], dtype=tf.float32)
        opponent_samples.append(opp_bids)
    opponent_samples = tf.convert_to_tensor(opponent_samples, dtype=tf.float32)
    
    if debug:
        print("opponent_samples generated: ", len(opponent_samples))
    
    # Step 2: Optimize agent's bid
    agent_bid = tf.Variable(tf.random.uniform([4, 2], maxval=[[1, 1], endowment, [1, 1], endowment]), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    for epoch in range(100):  # optimization steps
        with tf.GradientTape() as tape:
            total = 0
            # for opp_bids in opponent_samples:
            #     all_bids = tf.concat([opp_bids[:player_idx], tf.convert_to_tensor([agent_bid], dtype=tf.float32), opp_bids[player_idx:]], axis=0)
            #     _, utils, _, info = env.step(all_bids) # takes ~0.01s
            #     total += utils[player_idx]
            agent_bid_expanded = tf.expand_dims(tf.expand_dims(agent_bid, axis=0), axis=0)
            batched_bids = tf.tile(agent_bid_expanded, multiples=[opponent_samples.shape[0], 1, 1, 1])
            all_bids = tf.concat([opponent_samples[:, :player_idx], batched_bids, opponent_samples[:, player_idx:]], axis=1)
            _, utils, _, info = env.vector_step(all_bids, debug = debug) # takes ~0.015s
            total = tf.reduce_sum(utils[:, player_idx])
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

# === Meta-Game Construction ===
def build_meta_game(env: BidTradingEnv, policy_sets):
    joint_policies = list(itertools.product(*policy_sets))
    meta_game = []
    for i, joint in enumerate(joint_policies):
        payoff = env.evaluate_policy_tuple(joint)
        meta_game.append([joint, payoff])
    return meta_game

# === CCE Solver ===
def solve_cce(meta_game, num_players, policy_sets):
    joint_strats = [meta_game[i][0] for i in range(len(meta_game))]
    joint_payoffs = np.array([meta_game[i][1] for i in range(len(meta_game))])
    num_strats = len(joint_strats)
    
    sigma = cp.Variable(num_strats)
    objective = cp.Maximize(-0.5 * cp.quad_form(sigma, np.eye(num_strats)))
    constraints = [cp.sum(sigma) == 1, sigma >= 0]
    for p in range(num_players):
        for alt_pi in policy_sets[p]:
            constraint_sum = 0
            for i, pi in enumerate(joint_strats): # i is the index of the joint strategy, pi is the joint strategy
                pi_alt = list(pi)
                original_payoff = joint_payoffs[i][p] # get the payoff for the current strategy for player p
                if alt_pi.numpy().all() == pi[p].numpy().all(): # if the alternate strategy is the same as the current strategy, skip
                    alt_payoff = original_payoff
                else:
                    pi_alt[p] = alt_pi # update the current strategy for player p
                    if tuple(pi_alt) not in joint_strats: # if the alternate strategy is not in the meta game, skip
                        continue
                    alt_payoff = joint_payoffs[joint_strats.index(tuple(pi_alt))][p] # get the payoff for the alternate strategy for player p
                constraint_sum += sigma[i] * (alt_payoff - original_payoff) # add the constraint for the current strategy
            constraints.append(constraint_sum <= 1e-4) # add the constraint to the list of constraints
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return [(joint_strats[i], sigma.value[i]) for i in range(len(joint_strats))]


def run_training(debug = False, log_file = None):
    # === Main Loop ===
    num_players = 2
    obs_dim, goods, act_per_good = 2, 2, 4
    act_dim = goods * act_per_good
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    # optimal outcome is p=[1,1], x=[[6,2],[2,6]]
    env = BidTradingEnv(endowments, alphas)
    policy_sets = [[env.create_basic_policy(p)] for p in range(num_players)]
    meta_game = build_meta_game(env, policy_sets)
    sigma = solve_cce(meta_game, num_players, policy_sets)

    if debug:
        print("initial Configuration:")
        print("policy_sets: ", [[pi.numpy() for pi in policy_sets[p]] for p in range(num_players)])
        print("meta_game: ", [[i, payoff] for i, [joint, payoff] in enumerate(meta_game)])
        print("sigma: ", [[i, sigma[i][1]] for i in range(len(sigma))])
    elif log_file:
        log_file_path = log_file + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
        with open(log_file_path, "a") as f:
            f.write(f"Beginning Log for {log_file_path}. Training run.\n")
            f.write("ENVIRONMENT:\n")
            f.write(f"endowments:\n{np.array2string(endowments)}\n")
            f.write(f"alphas:\n{np.array2string(alphas)}\n")
            f.write("--------------------------------\n")
            f.write("Initial Configuration:\n")
            policy_sets_str = np.array2string(tf.convert_to_tensor(policy_sets).numpy())
            f.write(f"policy_sets: {policy_sets_str}\n")
            meta_game_str = np.array2string(tf.convert_to_tensor([payoff for i, [joint, payoff] in enumerate(meta_game)]).numpy())
            f.write(f"meta_game: {meta_game_str}\n")
            sigma_str = np.array2string(tf.convert_to_tensor([sigma[i][1] for i in range(len(sigma))]).numpy())
            f.write(f"sigma: {sigma_str}\n")
            f.write("--------------------------------\n")

    for epoch in range(50):
        for p in range(num_players):
            # train_best_response returns a tf.Variable, convert to numpy for logging
            new_pi_var = train_best_response(env, sigma, p, endowments[p], debug=debug)
            new_pi_np = new_pi_var.numpy() # Get numpy array from the Variable/Tensor

            if debug:
                print(f"new policy for player {p}:\n{new_pi_np}")
            elif log_file:
                with open(log_file_path, "a") as f:
                    f.write(f"Epoch {epoch}, Player {p} new policy:\n{str(new_pi_np)}\n")
            # Append the original tf.Variable/Tensor to policy_sets
            policy_sets[p].append(new_pi_var)

        meta_game = build_meta_game(env, policy_sets)
        sigma = solve_cce(meta_game, num_players, policy_sets)

        if not debug and log_file:
            with open(log_file_path, "a") as f:
                f.write("--------------------------------\n")
                f.write(f"End of epoch {epoch}:\n")
                policy_sets_str = np.array2string(tf.convert_to_tensor(policy_sets).numpy())
                f.write(f"policy_sets: {policy_sets_str}\n")
                meta_game_str = np.array2string(tf.convert_to_tensor([payoff for i, [joint, payoff] in enumerate(meta_game)]).numpy())
                f.write(f"meta_game: {meta_game_str}\n")
                sigma_str = np.array2string(tf.convert_to_tensor([sigma[i][1] for i in range(len(sigma))]).numpy())
                f.write(f"sigma: {sigma_str}\n")
                f.write("--------------------------------\n")

    # --- Reset NumPy print options if needed elsewhere ---
    # np.set_printoptions(threshold=1000, precision=8, suppress=False) # Reset to default or previous state

def test_env():
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    env = BidTradingEnv(endowments, alphas)
    bids_to_test = [tf.convert_to_tensor([[[1, 1], [1, 1], [1, 2], [1, 1]], [[1, 1], [1, 1], [2, 1], [1, 1]]], dtype=tf.float32), \
                    tf.convert_to_tensor([[[1, 1], [2, 0], [1, 1], [0, 2]], [[1, 1], [0, 2], [1, 1], [2, 0]]], dtype=tf.float32), \
                    tf.convert_to_tensor([[[1, 1], [2, 2], [1, 1], [-2, -2]], [[1, 1], [-2, -2], [1, 1], [2, 2]]], dtype=tf.float32)]
    for bid in bids_to_test:
        _, rewards, _, info = env.step(bid)
        print("rewards: ", [reward.numpy() for reward in rewards])
        print("executed_bids: ", info["executed_bids"].numpy())
        print("allocations: ", [[alloc.numpy() for alloc in allocation] for allocation in info["allocations"]])
        print("valid_bids: ", info["valid_bids"])
        print("net_credit: ", info["net_credit"])
        print("\n\n\n")

if __name__ == "__main__":
    run_training(debug = False, log_file = "oracle_log")
    #test_env()






'''
@tf.function
    def match_trades(self, bids):
        """
        Computes the net buy/sell quantities and average prices for each player and good.

        Args:
            bids: A tensor of shape (num_players, 4, num_goods), where:
                - bids[:, 0, g] = buy price for good g
                - bids[:, 1, g] = buy quantity for good g
                - bids[:, 2, g] = sell price for good g
                - bids[:, 3, g] = sell quantity for good g

        Returns:
            output_bids: A tensor of shape (num_players, 4, num_goods) with:
                - output[:, 0, g] = average buy price for good g
                - output[:, 1, g] = total bought quantity
                - output[:, 2, g] = average sell price
                - output[:, 3, g] = total sold quantity
        """
        num_players = tf.shape(bids)[0]
        num_goods = tf.shape(bids)[2]
        output_bids = tf.zeros((num_players, 4, num_goods), dtype=tf.float32)
        eps = tf.constant(1e-8, dtype=tf.float32)

        def weighted_avg_price(old_price, old_qty, new_price, new_qty):
            total_qty = old_qty + new_qty
            safe_qty = tf.maximum(total_qty, eps)
            return (old_price * old_qty + new_price * new_qty) / safe_qty

        def execute_trade(output, index, good, qty, price, is_buy):
            price_idx = 0 if is_buy else 2
            qty_idx = 1 if is_buy else 3

            old_qty = output[index, qty_idx, good]
            old_price = output[index, price_idx, good]

            new_qty = old_qty + qty
            avg_price = weighted_avg_price(old_price, old_qty, price, qty)

            output = tf.tensor_scatter_nd_update(output, [[index, qty_idx, good]], [new_qty])
            output = tf.tensor_scatter_nd_update(output, [[index, price_idx, good]], [avg_price])
            return output

        for good in tf.range(num_goods):
            buy_prices = bids[:, 0, good]
            sell_prices = bids[:, 2, good]

            buy_order = tf.argsort(buy_prices, direction="DESCENDING")
            sell_order = tf.argsort(sell_prices)

            buy_index = tf.constant(0)
            sell_index = tf.constant(0)

            buyer_idx = buy_order[buy_index]
            seller_idx = sell_order[sell_index]

            buy_price = tf.gather(bids[:, 0, good], buyer_idx)
            buy_qty = tf.gather(bids[:, 1, good], buyer_idx)
            sell_price = tf.gather(bids[:, 2, good], seller_idx)
            sell_qty = tf.gather(bids[:, 3, good], seller_idx)

            def loop_cond(output, bi, si, b_idx, s_idx, bp, bq, sp, sq):
                return tf.logical_and(
                    tf.logical_and(bi < num_players, si < num_players),
                    bp >= sp
                )

            def loop_body(output, bi, si, b_idx, s_idx, bp, bq, sp, sq):
                trade_qty = tf.minimum(bq, sq)
                should_trade = trade_qty > 0

                def trade():
                    out = execute_trade(output, b_idx, good, trade_qty, bp, True)
                    out = execute_trade(out, s_idx, good, trade_qty, bp, False)
                    return out

                output = tf.cond(should_trade, trade, lambda: output)

                bq = tf.cond(should_trade, lambda: bq - trade_qty, lambda: bq)
                sq = tf.cond(should_trade, lambda: sq - trade_qty, lambda: sq)

                next_bi = bi + 1
                has_next_buyer = next_bi < num_players
                next_buyer_idx = tf.cond(has_next_buyer, lambda: buy_order[next_bi], lambda: b_idx)
                advance_buyer = tf.logical_and(tf.equal(bq, 0), has_next_buyer)

                bi = tf.cond(tf.equal(bq, 0), lambda: next_bi, lambda: bi)
                b_idx = tf.cond(advance_buyer, lambda: next_buyer_idx, lambda: b_idx)
                bp = tf.cond(advance_buyer, lambda: tf.gather(bids[:, 0, good], next_buyer_idx), lambda: bp)
                bq = tf.cond(advance_buyer, lambda: tf.gather(bids[:, 1, good], next_buyer_idx), lambda: bq)

                next_si = si + 1
                has_next_seller = next_si < num_players
                next_seller_idx = tf.cond(has_next_seller, lambda: sell_order[next_si], lambda: s_idx)
                advance_seller = tf.logical_and(tf.equal(sq, 0), has_next_seller)

                si = tf.cond(tf.equal(sq, 0), lambda: next_si, lambda: si)
                s_idx = tf.cond(advance_seller, lambda: next_seller_idx, lambda: s_idx)
                sp = tf.cond(advance_seller, lambda: tf.gather(bids[:, 2, good], next_seller_idx), lambda: sp)
                sq = tf.cond(advance_seller, lambda: tf.gather(bids[:, 3, good], next_seller_idx), lambda: sq)

                return output, bi, si, b_idx, s_idx, bp, bq, sp, sq

            loop_vars = [output_bids, buy_index, sell_index, buyer_idx, seller_idx, buy_price, buy_qty, sell_price, sell_qty]
            output_bids, *_ = tf.while_loop(loop_cond, loop_body, loop_vars, maximum_iterations=num_players * 2 + 1)

        return output_bids
'''