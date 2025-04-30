import numpy as np
import tensorflow as tf
import json
from jpsro import BidTradingEnv, solve_cce

# This prevents NumPy from truncating large arrays with '...'
np.set_printoptions(threshold=np.inf, precision=4, suppress=True)

'''
This function tests the environment for the JPSRO. It tests the environment for the environment 
with 2 players, 2 goods, and set bids. It runs several bids and prints out the results to check 
to ensure that trading occurred properly.
'''
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

'''
This function tests the CCE meta-solver for the JPSRO. It tests the dominant strategy example
and the prof bryce example. In each one it creates a payoff tensor according to the example 
then passes it through the meta-solver to get the CCE

The first example is expected to have p=1 for R,R and p=0 for all other strategies.
The second example is expected to have L,L=0.17, L,R=0.0285, R,L=0.685, R,R=0.11
'''
def test_solver():
    L = tf.convert_to_tensor(np.array([1]))
    R = tf.convert_to_tensor(np.array([2]))
    meta_game_list = [[tuple([L, L]), [3, 3]], [tuple([L, R]), [1, 5]], [tuple([R, L]), [5, 6]], [tuple([R, R]), [7, 8]]]
    meta_game = [[joint, tf.convert_to_tensor(payoff)] for joint, payoff in meta_game_list]
    policy_sets = [[L, R], [L, R]]
    sigma = solve_cce(meta_game, 2, policy_sets)
    assert sigma[0][1] < 1e-4, "LL dominated, should have prob 0"
    assert sigma[1][1] < 1e-4, "LR dominated, should have prob 0"
    assert sigma[2][1] < 1e-4, "RL dominated, should have prob 0"
    assert sigma[3][1] > 1 - 1e-3, "RR dominates, should have prob 1"

    print("Dominant strategy example passed")

    meta_game_list = [[tuple([L, L]), [3, 4]], [tuple([L, R]), [1, 8]], [tuple([R, L]), [2, 6]], [tuple([R, R]), [7, 5]]]
    meta_game = [[joint, tf.convert_to_tensor(payoff)] for joint, payoff in meta_game_list]
    sigma = solve_cce(meta_game, 2, policy_sets)
    assert abs(sigma[0][1] - 0.1714) < 1e-2, "LL dominated, should have prob 0"
    assert abs(sigma[1][1] - 0.02858) < 1e-2, "LR dominated, should have prob 0"
    assert abs(sigma[2][1] - 0.6855) < 1e-2, "RL dominated, should have prob 0"
    assert abs(sigma[3][1] - 0.1144) < 1e-2, "RR dominates, should have prob 1"

    print("Prof Bryce example passed")

'''
Analyze the results of a JPSRO training run. takes in a results file which is expected 
to be in the format of the results file from the JPSRO training run: a json file with a 
list corresponding to the correlated equilibrium where each element is a list of two elements:
the first element is the joint policy and the second element is the probability of that joint policy.

This function will print the results in a readable format and assess the quality of the correlated equilibrium found
by verifying that it is a CCE then checking how close the strategies are to the optimal trading strategies.
'''
def analyze_results(res_file: str):
    num_players = 2
    num_goods = 2
    with open(res_file, "r") as f:
        results = json.load(f)
    print("there are ", len(results), " joint strategies")
    nonzero_results = [result for result in results if result[1] > 1e-4]
    print("there are ", len(nonzero_results), " joint strategies with probability > 1e-4")
    significant_results = [result for result in results if result[1] > 1e-2]
    print("there are ", len(significant_results), " joint strategies with probability > 0.01")
    
    max_prob_result = max(results, key=lambda x: x[1])
    all_max_prob_results = [result for result in results if max_prob_result[1] - result[1] < 1e-4]
    print("there are ", len(all_max_prob_results), " outcomes with the maximum probability")
    print("The maximum probability is:", max_prob_result[1])

    # set up the trading environment used for training
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    env = BidTradingEnv(endowments, alphas)

    _, rewards, _, info = env.step(tf.convert_to_tensor(max_prob_result[0], dtype=tf.float32))
    print("utility of max prob result: ", [reward.numpy() for reward in rewards])
    print("bid: \n", np.array(max_prob_result[0]))
    print("executed_bids: \n", info["executed_bids"].numpy())
    print("allocations: \n", [[alloc.numpy() for alloc in allocation] for allocation in info["allocations"]])
    print("valid_bids: \n", info["valid_bids"].numpy())
    print("net_credit: \n", info["net_credit"].numpy())

    print("\n--------------------------------\n")

    # compute average trade / utility
    all_outcomes = []
    avg_reward = np.zeros(num_players)
    avg_executed_trade = np.zeros((num_players, 4, num_goods))
    invalid_bid_count = np.zeros(num_players)
    invalid_bid_prob = np.zeros(num_players)
    net_credit_neg_count = np.zeros(num_players)
    net_credit_neg_prob = np.zeros(num_players)
    for result in results:
        _, rewards, _, info = env.step(tf.convert_to_tensor(result[0], dtype=tf.float32))
        all_outcomes.append({"rewards": rewards, "info": info, "prob": result[1]})
        avg_reward += np.array([reward.numpy() for reward in rewards]) * result[1]
        avg_executed_trade += info["executed_bids"].numpy() * result[1]
        invalid_bid_count += (info["valid_bids"].numpy() == 0)
        net_credit_neg_count += (info["net_credit"].numpy() < -0.25)
        invalid_bid_prob += (info["valid_bids"].numpy() == 0) * result[1]
        net_credit_neg_prob += (info["net_credit"].numpy() < -0.25) * result[1]
    print("average reward: ", avg_reward)
    print("average executed trade: ", avg_executed_trade)
    print("invalid bid count: ", invalid_bid_count)
    print("invalid bid prob: ", invalid_bid_prob)
    print("net credit neg count: ", net_credit_neg_count)
    print("net credit neg prob: ", net_credit_neg_prob)

    print("\n--------------------------------\n")

    # compute best trade / utility and percentage of time it is played
    best_outcome = max(all_outcomes, key=lambda x: tf.reduce_sum(x["rewards"]))
    print("utility of best outcome: ", [reward.numpy() for reward in best_outcome["rewards"]])
    print("executed bids: \n", best_outcome["info"]["executed_bids"].numpy())
    print("allocations: \n", [[alloc.numpy() for alloc in allocation] for allocation in best_outcome["info"]["allocations"]])
    print("valid_bids: \n", best_outcome["info"]["valid_bids"].numpy())
    print("net_credit: \n", best_outcome["info"]["net_credit"].numpy())
    print("probability of best outcome: ", best_outcome["prob"])

    print("\n--------------------------------\n")

    # make graph of cdf, where y axis is probability integral and x axis is utility

if __name__ == "__main__":
    #test_env()
    # test_solver()
    analyze_results("results/jprso_wrong_util/jprso_RMSprop_0.01_20_100_200.json")
