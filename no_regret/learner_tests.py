from learner import NoRegretLearner
import numpy as np
from typing import Dict, Callable, List
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import random

'''
This function checks if the given probabilities of each player playing actions 0,1 for a 
two player, bimatrix game of two actions comprise a CCE. It checks this by comparing the 
incentive constraints of the players, i.e. do they have an incentive to play a fixed strategy 
over the given probabilities.
'''
def check_is_cce(p_0, p_1, c_0, c_1, debug: bool = False):
    expected_cost_0 = c_0[0, 0] * p_0[0] * p_1[0] + \
                          c_0[0, 1] * p_0[0] * p_1[1] + \
                          c_0[1, 0] * p_0[1] * p_1[0] + \
                          c_0[1, 1] * p_0[1] * p_1[1]
        
    fixed_cost_0_0 = c_0[0, 0] * (p_0[0] * p_1[0] + p_0[1] * p_1[0]) + \
                        c_0[0, 1] * (p_0[0] * p_1[1] + p_0[1] * p_1[1])
    fixed_cost_0_1 = c_0[1, 0] * (p_0[0] * p_1[0] + p_0[1] * p_1[0]) + \
                        c_0[1, 1] * (p_0[0] * p_1[1] + p_0[1] * p_1[1])
    lowest_fixed_cost_0 = min(fixed_cost_0_0, fixed_cost_0_1)
    if debug:
        print(f"Player 0 expected cost: {expected_cost_0}")
        print(f"Player 0 lowest fixed cost: {lowest_fixed_cost_0}")
        #assert(expected_cost_0 <= lowest_fixed_cost_0), "Player 0 is not in expectation better than fixed strategy"

    # check if player 1 is in expectation better than fixed strategy
    expected_cost_1 = c_1[0, 0] * p_0[0] * p_1[0] + \
                        c_1[0, 1] * p_0[0] * p_1[1] + \
                        c_1[1, 0] * p_0[1] * p_1[0] + \
                        c_1[1, 1] * p_0[1] * p_1[1]
    
    fixed_cost_1_0 = c_1[0, 0] * (p_0[0] * p_1[0] + p_0[0] * p_1[1]) + \
                        c_1[1, 0] * (p_0[1] * p_1[0] + p_0[1] * p_1[1])
    fixed_cost_1_1 = c_1[0, 1] * (p_0[0] * p_1[0] + p_0[0] * p_1[1]) + \
                        c_1[1, 1] * (p_0[1] * p_1[0] + p_0[1] * p_1[1])
    lowest_fixed_cost_1 = min(fixed_cost_1_0, fixed_cost_1_1)
    if debug:
        print(f"Player 1 expected cost: {expected_cost_1}")
        print(f"Player 1 lowest fixed cost: {lowest_fixed_cost_1}")
        #assert(expected_cost_1 <= lowest_fixed_cost_1), "Player 1 is not in expectation better than fixed strategy"


'''
This function solves the CCE problem for a two player, bimatrix game of two actions.
It uses the linprog library to solve the linear programming problem, with constraints 
such that no player has an incentive to deviate to a fixed strategy from the given probabilities.
It also minimizes the sum of the expected cost of the players.
'''
def cce_solver(c_0, c_1, debug: bool = False):
    # expected cost vectors
    c_0_ex = c_0.flatten()
    c_1_ex = c_1.flatten()

    # constraints, for each player, the cost of playing a fixed strategy given the other player's strategy
    c_0_0 = np.array([c_0[0,0], c_0[0,1], c_0[0,0], c_0[0,1]])
    c_0_1 = np.array([c_0[1,0], c_0[1,1], c_0[1,0], c_0[1,1]])

    c_1_0 = np.array([c_1[0,0], c_1[0,0], c_1[1,0], c_1[1,0]])
    c_1_1 = np.array([c_1[0,1], c_1[0,1], c_1[1,1], c_1[1,1]])
    A_ub = np.array([c_0_ex - c_0_0, c_0_ex - c_0_1, c_1_ex - c_1_0, c_1_ex - c_1_1])
    b_ub = np.array([0, 0, 0, 0])
    # constraints, the sum of the probabilities of the players' strategies must be 1
    A_eq = np.array([[1, 1, 1, 1]])
    b_eq = np.array([1])
    # objective function, the sum of the expected cost of the players
    c = c_0_ex + c_1_ex
    # bounds, the probabilities must be between 0 and 1
    bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]
    # solve the linear programming problem
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    return res.x

'''
This function plots the changing probabilities of the players through the iterations of the no regret algorithm.
It plots the changing probabilities, the time averaged probabilities, and the joint time averaged probabilities.
'''
def show_plt(learner_0: NoRegretLearner, learner_1: NoRegretLearner, title: str):
    # Plot the changing probabilities through the iterations
    iterations = np.arange(len(learner_0.all_probabilities))
    plt.figure(figsize=(12, 6))
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.88)

    plt.subplot(2, 2, 1)
    plt.plot(iterations, learner_0.all_probabilities[:, 0], label='0, C')
    plt.plot(iterations, learner_0.all_probabilities[:, 1], label='0, D')
    plt.plot(iterations, learner_1.all_probabilities[:, 0], label='1, C')
    plt.plot(iterations, learner_1.all_probabilities[:, 1], label='1, D')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.title('Learner\'s Strategy Probabilities')
    plt.legend()

    learner_0_avg_prob = np.zeros((len(learner_0.all_probabilities), 2))
    learner_1_avg_prob = np.zeros((len(learner_1.all_probabilities), 2))
    for i in range(1, len(learner_0.all_probabilities)):
        learner_0_avg_prob[i, :] = np.mean(learner_0.all_probabilities[:i, :], axis=0)
        learner_1_avg_prob[i, :] = np.mean(learner_1.all_probabilities[:i, :], axis=0)

    plt.subplot(2, 2, 2)
    plt.plot(iterations, learner_0_avg_prob[:, 0], label='0, T')
    plt.plot(iterations, learner_0_avg_prob[:, 1], label='0, B')
    plt.plot(iterations, learner_1_avg_prob[:, 0], label='1, L')
    plt.plot(iterations, learner_1_avg_prob[:, 1], label='1, R')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.title('Learner\'s Time Average Strategy Probabilities')
    plt.legend()

    p_0_0 = np.zeros((len(learner_0.all_probabilities)))
    p_0_1 = np.zeros((len(learner_0.all_probabilities)))
    p_1_0 = np.zeros((len(learner_0.all_probabilities)))
    p_1_1 = np.zeros((len(learner_0.all_probabilities)))
    for i in range(1, len(learner_0.all_probabilities)):
        p_0_0[i] = np.mean(learner_0.all_probabilities[:i, 0]) * np.mean(learner_1.all_probabilities[:i, 0])
        p_0_1[i] = np.mean(learner_0.all_probabilities[:i, 0]) * np.mean(learner_1.all_probabilities[:i, 1])
        p_1_0[i] = np.mean(learner_0.all_probabilities[:i, 1]) * np.mean(learner_1.all_probabilities[:i, 0])
        p_1_1[i] = np.mean(learner_0.all_probabilities[:i, 1]) * np.mean(learner_1.all_probabilities[:i, 1])

    plt.subplot(2, 2, 3)
    plt.plot(iterations, p_0_0, label='T, L')
    plt.plot(iterations, p_0_1, label='T, R')
    plt.plot(iterations, p_1_0, label='B, L')
    plt.plot(iterations, p_1_1, label='B, R')
    plt.xlabel('Iterations')
    plt.ylabel('Probability')
    plt.title('Joint Strategy Probabilities')
    plt.legend()

    iterations = np.arange(len(learner_0.all_choices))

    p_0_0 = np.zeros((len(learner_0.all_choices)))
    p_0_1 = np.zeros((len(learner_0.all_choices)))
    p_1_0 = np.zeros((len(learner_0.all_choices)))
    p_1_1 = np.zeros((len(learner_0.all_choices)))
    for i in range(1, len(learner_0.all_choices)):
        p_0_0[i] = np.mean(np.array([1 if i == "C" or i == "L" else 0 for i in learner_0.all_choices[:i]]) * np.array([1 if i == "C" or i == "L" else 0 for i in learner_1.all_choices[:i]]))
        p_0_1[i] = np.mean(np.array([1 if i == "C" or i == "L" else 0 for i in learner_0.all_choices[:i]]) * np.array([1 if i == "D" or i == "R" else 0 for i in learner_1.all_choices[:i]]))
        p_1_0[i] = np.mean(np.array([1 if i == "D" or i == "R" else 0 for i in learner_0.all_choices[:i]]) * np.array([1 if i == "C" or i == "L" else 0 for i in learner_1.all_choices[:i]]))
        p_1_1[i] = np.mean(np.array([1 if i == "D" or i == "R" else 0 for i in learner_0.all_choices[:i]]) * np.array([1 if i == "D" or i == "R" else 0 for i in learner_1.all_choices[:i]]))

    plt.subplot(2, 2, 4)
    plt.plot(iterations, p_0_0, label='T, L')
    plt.plot(iterations, p_0_1, label='T, R')
    plt.plot(iterations, p_1_0, label='B, L')
    plt.plot(iterations, p_1_1, label='B, R')
    plt.xlabel('Iterations')
    plt.ylabel('Distribution over Plays')
    plt.title('Stochastic Joint Strategy Probabilities')
    plt.legend()

    plt.tight_layout()
    plt.show()

'''
This function runs the no regret algorithm for a two player, bimatrix game of two actions.
It samples the strategies of the players, then uses the cost matrix to calculate the cost of the players' strategies and updates them, 
then checks if the probabilities of the players have converged to the expected probabilities (if given).
'''
def run_no_regret_fixed_cost(learner_0: NoRegretLearner, learner_1: NoRegretLearner, get_cost_vector, expected_prob_0 = None, expected_prob_1 = None, debug: bool = False, max_iter: int = 10000, print_final: bool = True):
    def print_debug(count, strategy_0, strategy_1, cost_0, cost_1):
        print(f"Iteration {count}")
        print(f"P0 prob: {learner_0.probabilities}")
        print(f"P1 prob: {learner_1.probabilities}")
        print(f"P0 strategy: {strategy_0}")
        print(f"P1 strategy: {strategy_1}")
        print(f"P0 cost: {cost_0}")
        print(f"P1 cost: {cost_1}")
    # run no regret algorithm
    converged = False
    count = 0
    while not converged:
        # sample strategies
        strategy_0 = learner_0.sample_strategy()
        strategy_1 = learner_1.sample_strategy()
        # calculate cost vector
        cost_0 = get_cost_vector(0, [strategy_0, strategy_1])
        cost_1 = get_cost_vector(1, [strategy_0, strategy_1])
        if debug:
            print_debug(count, strategy_0, strategy_1, cost_0, cost_1)
        # update probabilities
        learner_0.update(cost_0)
        learner_1.update(cost_1)
        count += 1
        # check if the probabilities have converged
        if count % 100 == 0 and expected_prob_0 is not None and expected_prob_1 is not None:
            time_averaged_prob_0 = np.mean(learner_0.all_probabilities, axis=0)
            time_averaged_prob_1 = np.mean(learner_1.all_probabilities, axis=0)
            if np.allclose(time_averaged_prob_0, expected_prob_0, atol=0.01) and np.allclose(time_averaged_prob_1, expected_prob_1, atol=0.01):
                converged = True
        elif count > max_iter:
            break
    # return the time averaged probabilities
    time_averaged_prob_0 = np.mean(learner_0.all_probabilities, axis=0)
    time_averaged_prob_1 = np.mean(learner_1.all_probabilities, axis=0)
    if print_final:
        print("Final probabilities:")
        print(f"P0 prob: {time_averaged_prob_0}")
        print(f"P1 prob: {time_averaged_prob_1}")
    return count, converged


'''
This function runs the no regret algorithm for a two player, bimatrix game of two actions.
It uses the expected value of the other player's strategy, probabilistically, to calculate the cost of the players' strategies and updates them, 
then checks if the probabilities of the players have converged to the expected probabilities (if given).
'''
def run_no_regret_expected_cost(learner_0: NoRegretLearner, learner_1: NoRegretLearner, get_cost_vector, expected_prob_0 = None, expected_prob_1 = None, debug: bool = False, max_iter: int = 10000, print_final: bool = True):
    def print_debug(count, cost_0, cost_1):
        print(f"Iteration {count}")
        print(f"P0 prob: {learner_0.probabilities}")
        print(f"P1 prob: {learner_1.probabilities}")
        print(f"P0 cost: {cost_0}")
        print(f"P1 cost: {cost_1}")
    # run no regret algorithm
    converged = False
    count = 0
    while not converged:
        # sample strategies
        learner_0.sample_strategy()
        learner_1.sample_strategy()
        # calculate expected costs for each player and action
        cost_0 = get_cost_vector(0)
        cost_1 = get_cost_vector(1)
        if debug:
            print_debug(count, cost_0, cost_1)
        # update probabilities
        learner_0.update(cost_0)
        learner_1.update(cost_1)
        count += 1
        # check if the probabilities have converged
        if count % 100 == 0 and expected_prob_0 is not None and expected_prob_1 is not None:
            time_averaged_prob_0 = np.mean(learner_0.all_probabilities, axis=0)
            time_averaged_prob_1 = np.mean(learner_1.all_probabilities, axis=0)
            if np.allclose(time_averaged_prob_0, expected_prob_0, atol=0.01) and np.allclose(time_averaged_prob_1, expected_prob_1, atol=0.01):
                converged = True
        elif count > max_iter:
            break
    # return the time averaged probabilities
    time_averaged_prob_0 = np.mean(learner_0.all_probabilities, axis=0)
    time_averaged_prob_1 = np.mean(learner_1.all_probabilities, axis=0)
    if print_final:
        print("Final probabilities:")
        print(f"P0 prob: {time_averaged_prob_0}")
        print(f"P1 prob: {time_averaged_prob_1}")
    return count, converged

'''
Dominant strategy example where P(R,R) = 1
Each player submits a strategy, then the cost vector is calculated on the submitted strategies 
'''
def dominant_strategy_example_fixed_cost(debug: bool = False): 
    def get_cost_vector(player: int, strategies: List[str]):
        strategy_mapping = {"L": 0, "R": 1}
        if player == 0:
            # cost matrix, inverse of utility matrix u= [[3, 1], [5, 7]]
            c = np.array([[-3, -1], [-5, -7]])
            scaled_c = (c - np.min(c)) / (np.max(c) - np.min(c))
            if debug:
                print(f"Player 0 scaled cost matrix: {scaled_c}")
            # cost vector for all player 0 strategies, for given player 1 strategy
            return scaled_c[:, strategy_mapping[strategies[1]]]
        else:
            # cost matrix, inverse of utility matrix u= [[3, 5], [6, 8]]
            c = np.array([[-3, -5], [-6, -8]])
            scaled_c = (c - np.min(c)) / (np.max(c) - np.min(c))
            if debug:
                print(f"Player 1 scaled cost matrix: {scaled_c}")
            # cost vector for all player 1 strategies, for given player 0 strategy
            return scaled_c[strategy_mapping[strategies[0]], :]

    # initialize learners
    learner_0 = NoRegretLearner(["L", "R"])
    learner_1 = NoRegretLearner(["L", "R"])

    # run no regret algorithm
    count, converged = run_no_regret_fixed_cost(learner_0, learner_1, get_cost_vector, [0,1], [0,1], debug)
    show_plt(learner_0, learner_1, "Dominant strategy example, fixed cost")

    if not converged:
        assert(False), "No regret algorithm did not converge"
    else:
        print(f'Dominant strategy example passed in {count} iterations\n')


'''
Dominant strategy example where P(R,R) = 1
Each player submits a strategy, then the cost vector is calculated on the expected value of the strategies
'''
def dominant_strategy_example_expected_cost(debug: bool = False): 
    # initialize learners
    learner_0 = NoRegretLearner(["L", "R"])
    learner_1 = NoRegretLearner(["L", "R"])

    def get_cost_vector(player: int):
        if player == 0:
            # cost matrix, inverse of utility matrix u= [[3, 1], [5, 7]]
            c = np.array([[-3, -1], [-5, -7]])
            scaled_c = (c - np.min(c)) / (np.max(c) - np.min(c))
            if debug:
                print(f"Player 0 scaled cost matrix: {scaled_c}")
            # cost vector for all player 0 strategies, for probabilistic player 1 strategy
            return scaled_c[:, 0] * learner_1.probabilities[0] + scaled_c[:, 1] * learner_1.probabilities[1]
        else:
            # cost matrix, inverse of utility matrix u= [[3, 5], [6, 8]]
            c = np.array([[-3, -5], [-6, -8]])
            scaled_c = (c - np.min(c)) / (np.max(c) - np.min(c))
            if debug:
                print(f"Player 1 scaled cost matrix: {scaled_c}")
            # cost vector for all player 1 strategies, for probabilistic player 0 strategy
            return scaled_c[0, :] * learner_0.probabilities[0] + scaled_c[1, :] * learner_0.probabilities[1]    

    # run no regret algorithm
    count, converged = run_no_regret_expected_cost(learner_0, learner_1, get_cost_vector, [0,1], [0,1], debug, max_iter=2000)
    show_plt(learner_0, learner_1, "Dominant strategy example, expected cost")

    # if not converged:
    #     assert(False), "No regret algorithm did not converge"
    # else:
    #     print(f'Dominant strategy example passed in {count} iterations\n')


'''
Prof Bryce example where P(L,L) ~ 0.171, P(L,R) ~ 0.029, P(R,L) ~ 0.686, P(R,R) ~ 0.114
Each player submits a strategy, then the cost vector is calculated on the submitted strategies
'''
def prof_bryce_example_fixed_cost(debug: bool = False): 
    # cost matrix, inverse of utility matrix u= [[3, 1], [2, 7]]
    c_0 = np.array([[2/3, 1], [5/6, 0]])
    # cost matrix, inverse of utility matrix u= [[4, 8], [6, 5]]
    c_1 = np.array([[1, 0], [0.5, 0.75]])
    def get_player_cost(player: int, strategies: List[str]):
        strategy_mapping = {"L": 0, "R": 1}
        if player == 0: 
            # cost vector for all player 0 strategies, for given player 1 strategy
            return c_0[:, strategy_mapping[strategies[1]]]
        else: 
            # cost vector for all player 1 strategies, for given player 0 strategy
            return c_1[strategy_mapping[strategies[0]], :]

    # initialize learners
    learner_0 = NoRegretLearner(["L", "R"], initial_weights=np.array([1, 2]))
    learner_1 = NoRegretLearner(["L", "R"], initial_weights=np.array([2, 1]))

    # run no regret algorithm
    count, converged = run_no_regret_fixed_cost(learner_0, learner_1, get_player_cost, debug=debug, max_iter=2000)

    def check_is_cce():
        # get time averaged probabilities for each player
        p_0 = np.mean(learner_0.all_probabilities, axis=0)
        p_1 = np.mean(learner_1.all_probabilities, axis=0)
        return p_0, p_1

    p_0, p_1 = check_is_cce()
    p_dist = [float(p_0[0] * p_1[0]), float(p_0[0] * p_1[1]), float(p_0[1] * p_1[0]), float(p_0[1] * p_1[1])]
    print("Joint distribution:", p_dist)
    expected_cce = cce_solver(c_0, c_1)
    print("Expected CCE:", expected_cce)
    # check if the joint, time averaged probabilities match the expected CCE
    assert(np.allclose(p_dist, expected_cce, atol=0.05)), "Player 0 and Player 1 probabilities do not match expected CCE"
    print("Prof Bryce example passed\n")
    show_plt(learner_0, learner_1, "Prof Bryce example, fixed cost")


'''
Prof Bryce example where P(L,L) ~ 0.171, P(L,R) ~ 0.029, P(R,L) ~ 0.686, P(R,R) ~ 0.114
Cost vector is calculated on the expected value of the strategies for each player
'''
def prof_bryce_example_expected_cost(debug: bool = False): 
    # cost matrix, inverse of utility matrix u= [[3, 1], [2, 7]]
    c_0 = np.array([[2/3, 1], [5/6, 0]])
    # cost matrix, inverse of utility matrix u= [[4, 8], [6, 5]]
    c_1 = np.array([[1, 0], [0.5, 0.75]])
    def get_player_cost(player: int):
        if player == 0: 
            # cost vector for all player 0 strategies, for probabilistic player 1 strategy
            return c_0[:, 0] * learner_1.probabilities[0] + c_0[:, 1] * learner_1.probabilities[1]
        else: 
            # cost vector for all player 1 strategies, for probabilistic player 0 strategy
            return c_1[0, :] * learner_0.probabilities[0] + c_1[1, :] * learner_0.probabilities[1]

    # initialize learners
    learner_0 = NoRegretLearner(["L", "R"], initial_weights=np.array([1, 2]))
    learner_1 = NoRegretLearner(["L", "R"], initial_weights=np.array([2, 1]))

    # run no regret algorithm
    count, converged = run_no_regret_expected_cost(learner_0, learner_1, get_player_cost, debug=debug, max_iter=5000)

    def check_is_cce():
        # get time averaged probabilities for each player
        p_0 = np.mean(learner_0.all_probabilities, axis=0)
        p_1 = np.mean(learner_1.all_probabilities, axis=0)
        return p_0, p_1

    p_0, p_1 = check_is_cce()
    p_dist = [float(p_0[0] * p_1[0]), float(p_0[0] * p_1[1]), float(p_0[1] * p_1[0]), float(p_0[1] * p_1[1])]
    print("Joint distribution:", p_dist)
    expected_cce = cce_solver(c_0, c_1)
    print("Expected CCE:", expected_cce)
    # check if the joint, time averaged probabilities match the expected CCE
    assert(np.allclose(p_dist, expected_cce, atol=0.02)), "Player 0 and Player 1 probabilities do not match expected CCE"
    print("Prof Bryce example passed\n")
    show_plt(learner_0, learner_1, "Prof Bryce example, expected cost")


def game_of_chicken_example(debug: bool = False):
    # cost matrix, inverse of utility matrix u= [[0, 7], [2, 6]]
    c_0 = np.array([[1, 0], [5/7, 1/7]])
    # cost matrix, inverse of utility matrix u= [[0, 2], [7, 6]]
    c_1 = np.array([[1, 5/7], [0, 1/7]])
    def get_player_cost_fixed_strategy(player: int, strategies: List[str]):
        strategy_mapping = {"C": 0, "D": 1}
        if player == 0: 
            # cost vector for all player 0 strategies, for given player 1 strategy
            return c_0[:, strategy_mapping[strategies[1]]]
        else: 
            # cost vector for all player 1 strategies, for given player 0 strategy
            return c_1[strategy_mapping[strategies[0]], :]

    def get_player_cost_expected_strategy(player: int):
        if player == 0: 
            # cost vector for all player 0 strategies, for probabilistic player 1 strategy
            return c_0[:, 0] * learner_1.probabilities[0] + c_0[:, 1] * learner_1.probabilities[1]
        else: 
            # cost vector for all player 1 strategies, for probabilistic player 0 strategy
            return c_1[0, :] * learner_0.probabilities[0] + c_1[1, :] * learner_0.probabilities[1]

    # initialize learners
    learner_0 = NoRegretLearner(["C", "D"])
    learner_1 = NoRegretLearner(["C", "D"])

    # run no regret algorithm for fixed strategy learning
    count, converged = run_no_regret_fixed_cost(learner_0, learner_1, get_player_cost_fixed_strategy, debug=debug, max_iter=5000)
    
    # get time averaged probabilities for each player
    p_0 = np.mean(learner_0.all_probabilities, axis=0)
    p_1 = np.mean(learner_1.all_probabilities, axis=0)
    # get joint distribution of the two players
    p_dist = [float(p_0[0] * p_1[0]), float(p_0[0] * p_1[1]), float(p_0[1] * p_1[0]), float(p_0[1] * p_1[1])]
    # check if the joint, time averaged probabilities match the expected CCE
    check_is_cce(p_0, p_1, c_0, c_1, True)
    print("Joint distribution:", p_dist)
    expected_cce = cce_solver(c_0, c_1)
    print("Expected CCE:", expected_cce)
    show_plt(learner_0, learner_1, "Chicken example, fixed cost")
    print("\n")

    # re-initialize learners
    learner_0 = NoRegretLearner(["C", "D"], initial_weights=np.array([1, 1]))
    learner_1 = NoRegretLearner(["C", "D"], initial_weights=np.array([1, 1]))

    # run no regret algorithm
    count, converged = run_no_regret_expected_cost(learner_0, learner_1, get_player_cost_expected_strategy, debug=debug, max_iter=5000)
    # get time averaged probabilities for each player
    p_0 = np.mean(learner_0.all_probabilities, axis=0)
    p_1 = np.mean(learner_1.all_probabilities, axis=0)
    # get joint distribution of the two players
    p_dist = [float(p_0[0] * p_1[0]), float(p_0[0] * p_1[1]), float(p_0[1] * p_1[0]), float(p_0[1] * p_1[1])]
    # check if the joint, time averaged probabilities match the expected CCE
    check_is_cce(p_0, p_1, c_0, c_1, True)
    print("Joint distribution:", p_dist)
    expected_cce = cce_solver(c_0, c_1)
    print("Expected CCE:", expected_cce)
    show_plt(learner_0, learner_1, "Chicken example, expected cost")



def game_of_chicken_many_iterations(num_iterations = 20, debug: bool = False):
    def run_game(fixed_cost: bool = True):
        # cost matrix, inverse of utility matrix u= [[0, 7], [2, 6]]
        c_0 = np.array([[1, 0], [5/7, 1/7]])
        # cost matrix, inverse of utility matrix u= [[0, 2], [7, 6]]
        c_1 = np.array([[1, 5/7], [0, 1/7]])
        def get_player_cost_fixed_strategy(player: int, strategies: List[str]):
            strategy_mapping = {"C": 0, "D": 1}
            if player == 0: 
                # cost vector for all player 0 strategies, for given player 1 strategy
                return c_0[:, strategy_mapping[strategies[1]]]
            else: 
                # cost vector for all player 1 strategies, for given player 0 strategy
                return c_1[strategy_mapping[strategies[0]], :]

        def get_player_cost_expected_strategy(player: int):
            if player == 0: 
                # cost vector for all player 0 strategies, for probabilistic player 1 strategy
                return c_0[:, 0] * learner_1.probabilities[0] + c_0[:, 1] * learner_1.probabilities[1]
            else: 
                # cost vector for all player 1 strategies, for probabilistic player 0 strategy
                return c_1[0, :] * learner_0.probabilities[0] + c_1[1, :] * learner_0.probabilities[1]
        # initialize learners
        diff_weights = random.random() < 0.5
        if diff_weights:
            learner_0 = NoRegretLearner(["C", "D"], initial_weights=np.array([random.uniform(1, 2), random.uniform(1, 2)]))
            learner_1 = NoRegretLearner(["C", "D"], initial_weights=np.array([random.uniform(1, 2), random.uniform(1, 2)]))
        else:
            learner_0 = NoRegretLearner(["C", "D"])
            learner_1 = NoRegretLearner(["C", "D"])
        count = 0
        converged = False
        if fixed_cost:
            count, converged = run_no_regret_fixed_cost(learner_0, learner_1, get_player_cost_fixed_strategy, debug=debug, max_iter=10000, print_final=False)
        else:
            count, converged = run_no_regret_expected_cost(learner_0, learner_1, get_player_cost_expected_strategy, debug=debug, max_iter=10000, print_final=False)
        return count, converged, learner_0, learner_1, diff_weights


    nash_p1 = np.array([0, 0, 1, 0])
    nash_p2 = np.array([0, 1, 0, 0])
    nash_m1 = np.array([1/9, 2/9, 2/9, 4/9])
    cor_1 = np.array([0, 1/4, 1/4, 1/2])
    cor_2 = np.array([1/5, 2/5, 2/5, 0])
    counts = [0, 0, 0, 0, 0]

    # utility = np.array([0,9,9,12]).T
    # print("utility of nash p1:", nash_p1 @ utility)
    # print("utility of nash p2:", nash_p2 @ utility)
    # print("utility of nash m1:", nash_m1 @ utility)
    # print("utility of cor 1:", cor_1 @ utility)
    # print("utility of cor 2:", cor_2 @ utility)

    for i in range(num_iterations):
        count, converged, learner_0, learner_1, diff_weights = run_game(fixed_cost=True)
    
        # get time averaged probabilities for each player
        p_0 = np.mean(learner_0.all_probabilities, axis=0)
        p_1 = np.mean(learner_1.all_probabilities, axis=0)
        # get joint distribution of the two players
        p_dist = [float(p_0[0] * p_1[0]), float(p_0[0] * p_1[1]), float(p_0[1] * p_1[0]), float(p_0[1] * p_1[1])]
        
        if np.allclose(p_dist, nash_p1, atol=0.05):
            counts[0] += 1
        elif np.allclose(p_dist, nash_p2, atol=0.05):
            counts[1] += 1
        elif np.allclose(p_dist, nash_m1, atol=0.05):
            if diff_weights:
                print("Nash M1 with different weights")
            counts[2] += 1
        elif np.allclose(p_dist, cor_1, atol=0.05):
            counts[3] += 1
        elif np.allclose(p_dist, cor_2, atol=0.05):
            counts[4] += 1
        else:
            print("WARNING: No match found:", p_dist)
        # print(f"Iteration {i+1}: {counts}")


    print(f"Nash P1: {counts[0]/num_iterations}")
    print(f"Nash P2: {counts[1]/num_iterations}")
    print(f"Nash M1: {counts[2]/num_iterations}")
    print(f"Cor 1: {counts[3]/num_iterations}")
    print(f"Cor 2: {counts[4]/num_iterations}")

 

if __name__ == "__main__":
    print("RUNNING NO REGRET LEARNER TESTS...\n\n")

    """
    TESTING EXAMPLES FOR 2 PLAYERS, 2 STRATEGIES
    """
    # dominant strategy example where P(R,R) = 1
    # the utility matrixes are u_0 = [[3, 1], [5, 7]], u_1 = [[3, 5], [6, 8]]
    print("Testing dominant strategy example...")
    # run no regret algorithm where all agents submit strategies then the cost vector is calculated on the submitted strategies
    dominant_strategy_example_fixed_cost(False)
    # # run no regret algorithm where the cost vector is calculated on the expected value of the strategies
    dominant_strategy_example_expected_cost(False)

    # example from youtube video on CE/CCE where the correlated equilibrium is a mixed strategy, 
    # P(L,L) ~ 0.171, P(L,R) ~ 0.029, P(R,L) ~ 0.686, P(R,R) ~ 0.114
    # the utility matrixes are u_0 = [[3, 1], [2, 7]], u_1 = [[4, 8], [6, 5]]
    # print("Testing prof bryce example...")
    prof_bryce_example_fixed_cost(False)
    prof_bryce_example_expected_cost(False)

    # chicken example where the mixed Nash equilibrium is P(C,C) ~ 0.17, P(C,D) ~ 0.03, P(D,C) ~ 0.69, P(D,D) ~ 0.11
    # and there exists two pure Nash equilibria, (C,D) and (D,C)
    # the utility matrixes are u_0 = [[0, 7], [2, 6]], u_1 = [[0, 2], [7, 6]]
    # print("Testing chicken example...")
    game_of_chicken_example(False)

    # run game of chicken example with many iterations
    # game_of_chicken_many_iterations(50, False)

