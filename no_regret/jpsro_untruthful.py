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
from jpsro import BidTradingEnv, build_meta_game, solve_cce, train_best_response, prune_strategies, PARAM_DICT


'''
    Wrapper on the bid trading environment to allow for switching between truthful and false best responding
    Using to see if agents behaviour changes if they misrepresent their utilities in the BR selection
'''
class UntruthfulBidTradingEnv(BidTradingEnv):
    def __init__(self, endowments, true_cobb_douglas_exponents, false_cobb_douglas_exponents):
        super().__init__(endowments, true_cobb_douglas_exponents)
        self.true_alphas = tf.convert_to_tensor(true_cobb_douglas_exponents, dtype=tf.float32)
        self.false_alphas = tf.convert_to_tensor(false_cobb_douglas_exponents, dtype=tf.float32)
    
    def switch_utilities(self, truthful: bool):
        if truthful:
            self.alphas = self.true_alphas
        else:
            self.alphas = self.false_alphas
        return self.alphas


'''
    Wrapper on the meta-game step. Make sure the utilities are switched to the true values
'''
def build_meta_game_true(env: UntruthfulBidTradingEnv, policy_sets):
    env.switch_utilities(True)
    return build_meta_game(env, policy_sets)

'''
    Wrapper on the BR step. Make sure the utilities are switched to the untuthful values
'''
def train_best_response_false(env, joint_distribution, player_idx, endowment, debug=False):
    env.switch_utilities(False)
    return train_best_response(env, joint_distribution, player_idx, endowment, debug)

'''
    Run the JPSRO algorithm. This is done by building the meta-game, solving the CCE, and then using the best response to add a new policy
    to the policy set. This is done iteratively until the policy sets converge.

    Every pruning_parameter steps, the policy sets are pruned to remove strategies that are not played.

    Same algorithm as in jpsro.py except using the wrapper functions above
'''
def jprso(env: UntruthfulBidTradingEnv, debug = False, log_file = None):
    # Initialize the algorithm and create step 0 meta-game and CCE
    num_players = env.num_players
    policy_sets = [[env.create_basic_policy(p)] for p in range(num_players)]
    meta_game = build_meta_game_true(env, policy_sets)
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
            new_pi_var = train_best_response_false(env, sigma, p, env.endowments[p], debug=debug)
            new_pi_np = new_pi_var.numpy() # Get numpy array from the Variable/Tensor

            if debug:
                print(f"new policy for player {p}:\n{new_pi_np}")
            elif log_file:
                with open(log_file, "a") as f:
                    f.write(f"Epoch {epoch}, Player {p} new policy:\n{str(new_pi_np)}\n")
            # Append the original tf.Variable/Tensor to policy_sets
            policy_sets[p].append(new_pi_var)

        # build the new meta-game and solve the new CCE
        meta_game = build_meta_game_true(env, policy_sets)
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
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    false_alphas = np.array([[0.75, 0.25], [0.75, 0.25]])
    # optimal outcome is p=[1,1], x=[[6,2],[2,6],[5.33,2.67],[2.67,5.33]]
    env = UntruthfulBidTradingEnv(endowments, alphas, false_alphas)

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

if __name__ == "__main__":
    log_prefix = f"jpsro_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"
    os.makedirs(f"logs/{log_prefix}", exist_ok=True)
    os.makedirs(f"results/{log_prefix}", exist_ok=True)
    log_filename = "jprso_untruthful"
    sigma = run_training(debug = False, log_file = f"logs/{log_prefix}{log_filename}.txt")
    sigma_processed = [[[strat.numpy().tolist() for strat in joint], prob] for joint, prob in sigma]
    with open(f"results/{log_prefix}{log_filename}.json", "a") as f:
        f.write(f"{json.dumps(sigma_processed)}\n")