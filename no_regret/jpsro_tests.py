import numpy as np
import tensorflow as tf
import json
import os
import matplotlib.pyplot as plt
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
        all_outcomes.append({"bids": result[0], "rewards": rewards, "info": info, "prob": result[1]})
        avg_reward += np.array([reward.numpy() for reward in rewards]) * result[1]
        avg_executed_trade += info["executed_bids"].numpy() * result[1]
        invalid_bid_count += (info["valid_bids"].numpy() == 0)
        net_credit_neg_count += (info["net_credit"].numpy() < -0.25)
        invalid_bid_prob += (info["valid_bids"].numpy() == 0) * result[1]
        net_credit_neg_prob += (info["net_credit"].numpy() < -0.25) * result[1]
    print("average reward: ", avg_reward)
    print("average executed trade: \n", avg_executed_trade)
    print("invalid bid count: ", invalid_bid_count)
    print("invalid bid prob: ", invalid_bid_prob)
    print("net credit neg count: ", net_credit_neg_count)
    print("net credit neg prob: ", net_credit_neg_prob)

    print("\n--------------------------------\n")

    # compute best trade / utility and percentage of time it is played
    best_outcome = max(all_outcomes, key=lambda x: tf.reduce_sum(x["rewards"]))
    print("utility of best outcome: ", [reward.numpy() for reward in best_outcome["rewards"]])
    print("bids: \n", np.array(best_outcome["bids"]))
    print("executed bids: \n", best_outcome["info"]["executed_bids"].numpy())
    print("allocations: \n", [[alloc.numpy() for alloc in allocation] for allocation in best_outcome["info"]["allocations"]])
    print("valid_bids: \n", best_outcome["info"]["valid_bids"].numpy())
    print("net_credit: \n", best_outcome["info"]["net_credit"].numpy())
    print("probability of best outcome: ", best_outcome["prob"])

    print("\n--------------------------------\n")

    # make graph of cdf, where y axis is probability integral and x axis is utility

    # Calculate the sum of utilities for each outcome
    total_utilities = [tf.reduce_sum(outcome["rewards"]).numpy() for outcome in all_outcomes]
    p1_utilities = [outcome["rewards"][0].numpy() for outcome in all_outcomes]
    p2_utilities = [outcome["rewards"][1].numpy() for outcome in all_outcomes]

    # Sort outcomes by utility
    sorted_outcomes = sorted(zip(total_utilities, all_outcomes), key=lambda x: x[0])
    sorted_p1_outcomes = sorted(zip(p1_utilities, all_outcomes), key=lambda x: x[0])
    sorted_p2_outcomes = sorted(zip(p2_utilities, all_outcomes), key=lambda x: x[0])

    # Calculate cumulative probabilities for total utilities
    cumulative_probabilities_total = []
    cumulative_prob_total = 0
    for _, outcome in sorted_outcomes:
        cumulative_prob_total += outcome["prob"]
        cumulative_probabilities_total.append(cumulative_prob_total)

    # Extract sorted total utilities for plotting
    sorted_total_utilities = [utility for utility, _ in sorted_outcomes]

    # Calculate cumulative probabilities for player 1 utilities
    cumulative_probabilities_p1 = []
    cumulative_prob_p1 = 0
    for _, outcome in sorted_p1_outcomes:
        cumulative_prob_p1 += outcome["prob"]
        cumulative_probabilities_p1.append(cumulative_prob_p1)

    # Extract sorted player 1 utilities for plotting
    sorted_p1_utilities = [utility for utility, _ in sorted_p1_outcomes]

    # Calculate cumulative probabilities for player 2 utilities
    cumulative_probabilities_p2 = []
    cumulative_prob_p2 = 0
    for _, outcome in sorted_p2_outcomes:
        cumulative_prob_p2 += outcome["prob"]
        cumulative_probabilities_p2.append(cumulative_prob_p2)

    # Extract sorted player 2 utilities for plotting
    sorted_p2_utilities = [utility for utility, _ in sorted_p2_outcomes]

    # Plot the CDFs in a grid
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Plot total utilities CDF
    axs[0].plot(sorted_total_utilities, cumulative_probabilities_total, marker='o')
    axs[0].set_title('Total Welfare CDF')
    axs[0].set_xlabel('Total Welfare of the Bid')
    axs[0].set_ylabel('Cumulative Probability')
    axs[0].grid(True)

    # Plot player 1 utilities CDF
    axs[1].plot(sorted_p1_utilities, cumulative_probabilities_p1, marker='o', color='orange')
    axs[1].set_title('Player 1 Utility CDF')
    axs[1].set_xlabel('Player 1 Utility')
    axs[1].set_ylabel('Cumulative Probability')
    axs[1].grid(True)

    # Plot player 2 utilities CDF
    axs[2].plot(sorted_p2_utilities, cumulative_probabilities_p2, marker='o', color='green')
    axs[2].set_title('Player 2 Utility CDF')
    axs[2].set_xlabel('Player 2 Utility')
    axs[2].set_ylabel('Cumulative Probability')
    axs[2].grid(True)

    plt.suptitle(f'Cumulative Distribution Functions of Outcomes\n{file.split(".js")[0]}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    #test_env()
    # test_solver()
    directory = "results/jprso_wrong_util"
    files_in_directory = os.listdir(directory)
    for file in files_in_directory:
        print(file)
        analyze_results(directory + "/" + file)
