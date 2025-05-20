import numpy as np
import tensorflow as tf
import json
import os
import csv
import matplotlib.pyplot as plt
from jpsro import BidTradingEnv, solve_cce

'''
    This file contains the tests for the JPSRO algorithm.
'''

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
    This function takes in a results file and prints the stats for the results.
'''
def get_file_stats(file: str):
    num_players = 2
    num_goods = 2
    with open(file, "r") as f:
        results = json.load(f)
    nonzero_results = [result for result in results if result[1] > 1e-4]
    significant_results = [result for result in results if result[1] > 1e-2]
    
    max_prob_result = max(results, key=lambda x: x[1])
    all_max_prob_results = [result for result in results if max_prob_result[1] - result[1] < 1e-4]

    # set up the trading environment used for training
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    env = BidTradingEnv(endowments, alphas)

    _, rewards, _, info = env.step(tf.convert_to_tensor(max_prob_result[0], dtype=tf.float32))

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
    
    return {"avg_reward": avg_reward, "avg_executed_trade": avg_executed_trade, \
            "invalid_bid_count": invalid_bid_count, "invalid_bid_prob": invalid_bid_prob, \
            "net_credit_neg_count": net_credit_neg_count, "net_credit_neg_prob": net_credit_neg_prob, \
            "max_prob_result": max_prob_result, "all_max_prob_results": all_max_prob_results, \
            "num_strats": len(results), "num_nonzero_strats": len(nonzero_results), \
            "num_significant_strats": len(significant_results), "num_all_max_prob_strats": len(all_max_prob_results)}

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

    plt.suptitle(f'Cumulative Distribution Functions of Outcomes\n{res_file.split(".js")[0]}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


'''
    This function plots the average reward vs. learning rate for a given directory.
'''
def plot_optimizer_sweep(directory: str):
    files_in_directory = os.listdir(directory)
    file_map = []
    for file in files_in_directory:
        file_name = file.split(".js")[0]
        file_name = file_name.split("_")
        optimizer = file_name[1]
        learning_rate = file_name[2]
        max_epochs = file_name[3]
        best_response_epochs = file_name[4]
        best_response_samples = file_name[5]
        file_map.append({"file": file, "optimizer": optimizer, "learning_rate": learning_rate, "max_epochs": max_epochs, "best_response_epochs": best_response_epochs, "best_response_samples": best_response_samples})
    
    optimizer_data = {}
    for file in file_map:
        stats = get_file_stats(directory + "/" + file["file"])
        avg_reward = np.sum(stats["avg_reward"])  # Calculate the mean of average rewards
        optimizer = file["optimizer"]
        learning_rate = float(file["learning_rate"])

        if optimizer not in optimizer_data:
            optimizer_data[optimizer] = {"learning_rates": [], "avg_rewards": []}

        optimizer_data[optimizer]["learning_rates"].append(learning_rate)
        optimizer_data[optimizer]["avg_rewards"].append(avg_reward)

    plt.figure(figsize=(15, 9))
    # Sort the data by learning rate for each optimizer
    for optimizer, data in optimizer_data.items():
        sorted_indices = np.argsort(data["learning_rates"])
        sorted_learning_rates = np.array(data["learning_rates"])[sorted_indices]
        sorted_avg_rewards = np.array(data["avg_rewards"])[sorted_indices]
        
        # Plot average reward vs. learning rate for each optimizer
        plt.plot(sorted_learning_rates, sorted_avg_rewards, marker='o', label=optimizer)

    plt.title('Average Reward vs. Learning Rate')
    plt.xlabel('Learning Rate')
    plt.ylabel('Average Reward')
    plt.xscale('log')  # Use logarithmic scale for learning rate
    plt.legend(title='Optimizer')
    plt.grid(True)
    plt.show()

'''
    This function plots the average reward vs. max epochs for a given directory.
'''
def plot_parameter_sweep(directory: str):
    files_in_directory = os.listdir(directory)
    file_map = []
    for file in files_in_directory:
        file_name = file.split(".js")[0]
        file_name = file_name.split("_")
        optimizer = file_name[1]
        learning_rate = file_name[2]
        max_epochs = file_name[3]
        best_response_epochs = file_name[4]
        best_response_samples = file_name[5]
        file_map.append({"file": file, "optimizer": optimizer, "learning_rate": learning_rate, "max_epochs": max_epochs, "best_response_epochs": best_response_epochs, "best_response_samples": best_response_samples})
    
    max_epochs_data = {}
    for file in file_map:
        learning_rate = float(file["learning_rate"])
        if learning_rate == 0.005:
            continue  # Ignore trials with learning_rate=0.005

        stats = get_file_stats(directory + "/" + file["file"])
        avg_reward = np.sum(stats["avg_reward"])  # Calculate the mean of average rewards
        max_epochs = int(file["max_epochs"])
        best_response_epochs = file["best_response_epochs"]
        best_response_samples = file["best_response_samples"]

        # Create a unique key for the combination of best_response_epochs and best_response_samples
        key = (best_response_epochs, best_response_samples)

        if key not in max_epochs_data:
            max_epochs_data[key] = {"max_epochs": [], "avg_rewards": []}

        max_epochs_data[key]["max_epochs"].append(max_epochs)
        max_epochs_data[key]["avg_rewards"].append(avg_reward)

    plt.figure(figsize=(8, 4))
    # Sort the data by max_epochs for each combination of best_response_epochs and best_response_samples
    for key, data in max_epochs_data.items():
        sorted_indices = np.argsort(data["max_epochs"])
        sorted_max_epochs = np.array(data["max_epochs"])[sorted_indices]
        sorted_avg_rewards = np.array(data["avg_rewards"])[sorted_indices]
        
        # Plot average reward vs. max_epochs for each combination
        plt.plot(sorted_max_epochs, sorted_avg_rewards, marker='o', label=f"BR Epochs: {key[0]}, BR Samples: {key[1]}")

    plt.title('Average Reward vs. Max Epochs')
    plt.xlabel('Max Epochs')
    plt.ylabel('Average Reward')
    plt.legend(title='BR Epochs & Samples')
    plt.grid(True)
    plt.show()

'''
    This function writes the stats for a given directory to a csv file.
'''
def reward_to_csv(directory: str):
    files_in_directory = os.listdir(directory)
    file_map = []
    for file in files_in_directory:
        file_name = file.split(".js")[0]
        file_name = file_name.split("_")
        optimizer = file_name[1]
        learning_rate = file_name[2]
        max_epochs = file_name[3]
        best_response_epochs = file_name[4]
        best_response_samples = file_name[5]
        file_map.append({"file": file, "optimizer": optimizer, "learning_rate": learning_rate, "max_epochs": max_epochs, "best_response_epochs": best_response_epochs, "best_response_samples": best_response_samples})

    # Get the stats for each file and write to a CSV
    csv_filename = "reward_stats.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['max_epochs', 'best_response_epochs', 'best_response_samples', 
                    'p1 avg. utility', 'p2 avg. utility', 'average utility sum']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        stats_list = []
        for file in file_map:
            learning_rate = float(file["learning_rate"])
            if learning_rate == 0.005:
                continue  # Exclude learning rate 0.005

            stats = get_file_stats(directory + "/" + file["file"])
            avg_reward = stats["avg_reward"]
            avg_utility_sum = np.sum(avg_reward)
            max_epochs = int(file["max_epochs"])
            best_response_epochs = file["best_response_epochs"]
            best_response_samples = file["best_response_samples"]

            stats_list.append({
                'max_epochs': max_epochs,
                'best_response_epochs': best_response_epochs,
                'best_response_samples': best_response_samples,
                'p1 avg. utility': round(avg_reward[0], 3),
                'p2 avg. utility': round(avg_reward[1], 3),
                'average utility sum': round(avg_utility_sum, 3)
            })

        # Sort the stats by average utility sum
        stats_list.sort(key=lambda x: x['average utility sum'], reverse=True)

        # Write sorted stats to CSV
        for stats in stats_list:
            writer.writerow(stats)
    


if __name__ == "__main__":
    test_env()
    # test_solver()
    # directory = "results/jpsro_param"
    # files_in_directory = ["jprso_RMSprop_0.001_30_200_100.json","jprso_RMSprop_0.001_50_50_500.json","jprso_RMSprop_0.001_50_200_200.json","jprso_RMSprop_0.001_30_500_200.json","jprso_RMSprop_0.001_30_100_200.json"] # os.listdir(directory)
    # for file in files_in_directory:
    #     print(file)
    #     analyze_results(directory + "/" + file)
    #plot_optimizer_sweep("results/jpsro_MWCCE")
    # reward_to_csv("results/jpsro_param")
