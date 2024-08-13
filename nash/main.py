import random
import numpy as np
from scipy.optimize import linprog
EPS = 1e-5
# Reference notes for this code can be found at https://www.notion.so/Summer-2024-Notes-b6100cca39664b20b6f53d51b847e80c?pvs=4
###

###
# INPUTS
# S: the matrix of strategies for each player n, where S[n] is the list of strategies for player n
# n: the player to calculate utility for
# i: the strategy of n the expectation is conditional on
#
# RETURNS
# an array of all plays of the game where player n plays strategy i, each play is an array of size N where N is the number of players, 
# and each element of the play is an array of size S[n] with exactly one element equal to 1
###
def get_all_plays(S, n, i):
    plays = []
    def generate_plays(S, current_play, row, n, i):
        if row == n:
            if n == len(S) - 1:
                plays.append(current_play)
                return
            else: 
                generate_plays(S, current_play, row + 1, n, i)
                return
        elif row ==len(S) - 1:
            for j in range(len(S[row])):
                new_play = [r[:] for r in current_play]
                new_play[row][j] = 1
                plays.append(new_play)
            return          
        for j in range(len(S[row])):
            new_play = [r[:] for r in current_play]
            new_play[row][j] = 1
            generate_plays(S, new_play, row + 1, n, i)

    initial_play = [[0] * len(row) for row in S]
    initial_play[n][i] = 1
    generate_plays(S, initial_play, 0, n, i)
    return plays

# Let x be a mixed strategy profile of the game, i.e. the same dimensionality of w except each row of x (sum over j for x[i][j]) = 1
# A[n][i](x) = sum over all (pure) strategy plays w where w[n][i]=1(-u[n](w) * product over all other players v(not n)(x[v][j])), where j is the entry of w for player v that is 1
# 
# INPUTS
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
# u_n: the payoff function for player n
# n: the player to calculate utility for
# i: the strategy of n the expectation is conditional on
#
# RETURNS
# the scalar value of the opposite of the conditional expected utility for player n playing i given strategy profile x 
###
def neg_conditional_expected_utility(x, u_n, n, i):
    utility_sum = 0
    for w in get_all_plays(S, n, i):
        a = -(u_n(w))
        other_player_product = 1
        for v in range(len(x)):
            if v != n:
                j = w[v].index(1)
                other_player_product *= x[v][j]
        utility_sum += a * other_player_product
    return utility_sum

###
# INPUTS
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
# u_n: the payoff function for player n
# n: the player to calculate utility for
#
# RETURNS
# the scalar value of the opposite of the conditional expected utility for player n given strategy profile x 
###
def neg_expected_utility(x, u_n, n):
    utility_sum = 0
    # calculate sum of conditional utility times the probability of that outcome for each strategy for player n
    for i in range(len(x[n])):
        conditional_sum = neg_conditional_expected_utility(x, u_n, n, i)
        utility_sum += conditional_sum * x[n][i]
    return utility_sum

###
# INPUTS
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
#
# RETURNS
# x*, the same mixed strategy profile s.t. x[n][i]/x[n][j] = x*[n][i]/x*[n][j] and sum(x[n]) = 1
###
def normalize_strategy_profile(x):
    normalized_x = []
    for player in x:
        total = sum(player)
        if total == 0:
            raise ValueError("Strategy profile cannot be normalized because a player has no strategies with nonzero probability")
        else:
            normalized_x.append([s / total for s in player])
    return normalized_x

###
# INPUTS
# S: the matrix of strategies for each player n, where S[n] is the list of strategies for player n
# 
# RETURNS
# x: an array of size S where exactly one element per row = 1, all else are 0
###
def fix_all_strategies(S):
    x = []
    for strategies in S:
        player_strategy = [0] * len(strategies)
        random_strategy = random.choice(range(len(strategies)))
        player_strategy[random_strategy] = 1
        x.append(player_strategy)
    return x

###
# INPUTS
# S: the matrix of strategies for each player n, where S[n] is the list of strategies for player n
# u_n: the payoff function for player n
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
# n: the player to calculate the best response for
#
# RETURNS
# the mixed strategy profile for player n that is a best response to all x[v] for v != n 
# result is returned as a one dimensional array of length S[n]
###
def best_mixed_response(S, u_n, x, n):
    num_strategies = len(S[n])
    c = [neg_conditional_expected_utility(x, u_n, n, i) for i in range(num_strategies)]  # Objective function (maximize utility)
    # Sum of probabilities must be 1, linprog calculates in the form A_eq * x = b_eq
    A_eq = [[1] * num_strategies]
    b_eq = [1]

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,1), method='highs')
    
    if result.success:
        return result.x  # Optimal mixed strategy profile for player n
    else:
        raise ValueError("Linear programming failed to find a solution", str(result.status))
    
###
# INPUTS
# u_n: the payoff function for player n
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
# n: the player to calculate the boundary conditions for
# i: the strategy of n to calculate the boundary conditions for
#
# RETURNS
# a dictionary with keys 'x' and 'y' and boolean values for whether the boundary conditions are satisfied for each value
###
def calculate_boundary_conditions(u_n, x, n, i):
    boundary_x = abs(x[n][i]) #< EPS
    boundary_y = abs(neg_conditional_expected_utility(x, u_n, n, i) - 1) #< EPS
    return {'x': boundary_x, 'y': boundary_y}

###
# INPUTS
# U: array of payoff functions for each player, array where u[n] is the payoff function for player n
# x: the mixed strategy profile, array dimension N, each element i an array of length S[i]
# N: the number of players to calculate boundary conditions for (with non-fixed strategies)
#
# RETURNS
# an array of the form [boundary_conditions, num_conditions_satisfied, complementary]
# where boundary_conditions[n][i] = {'x': boundary_x, 'y': boundary_y} (boolean values true iff x[n][i] or y[n][i] is within epsilon of 0),
# num_conditions_satisfied is the number of boundary conditions that are satisfied across all n,i,
# complementary is a boolean that is true if the boundary conditions signal a complementary node
###
def calculate_all_boundary_conditions(U, x, N):
    K = sum(len(S[i]) for i in range(N))
    boundary_conditions = []
    num_conditions_satisfied = 0
    complementary = True
    for n in range(N):
        player_conditions = []
        for i in range(len(x[n])):
            conditions = calculate_boundary_conditions(U[n], x, n, i)
            player_conditions.append(conditions)
            if conditions['x'] < EPS:
                num_conditions_satisfied += 1
                if conditions['y'] < EPS:
                    num_conditions_satisfied += 1
                    complementary = False
            elif conditions['y'] < EPS:
                num_conditions_satisfied += 1
        boundary_conditions.append(player_conditions)
    complementary = complementary and num_conditions_satisfied == K
    return boundary_conditions, num_conditions_satisfied, complementary


### Problem Definition
# INPUTS
# N: number of players
# S: list of pure strategies for each player, array where S[n] is the list of strategies for player n
# U: array of payoff functions for each player, array where u[n] is the payoff function for player n
#   u[n] is a function that takes a play w of the game of the form of an array of size N x S[i], where N is the number of players and S[i] is the set of strategies for player i
#        the sum of each row of w is 1 where w[i][j] is 1 for exactly 1 j
# CALCULATE
# K = the dimensionality of the strategy space = sum(len(S[i]) for i in range(n))
# Let x be a mixed strategy profile of the game, i.e. the same dimensionality of w except each row of x (sum over j for x[i][j]) = 1
# A[n][i](x) = sum over all (pure) strategy plays w where w[n][i]=1(-u[n](w) * product over all other players v(not n)(x[v][j])), where j is the entry of w for player v that is 1
# y[n][i] = A[n][i] - 1
# STRATEGY
# Fix all strategies for players n > 1 randomly
# Find optimal strategy for that 1 player game
# Free another player strategy and traverse:
#   starting at a node, it must satisfy K boundary conditions, meaning either it is already an equilibrium (if K conditions are each one of x[n][i] or y[n][i]), or there is some 
#   n, i, s.t. x[n][i] = 0 and y[n][i] = 0, which is a (m,j)-almost-complementary node
#   find this n, i, then increment x[n][i] by epsilon-small intervals and check if K conditions are again satisfied (will otherwise be K-1)
#   satisfied here means that within epsilon of 0
#   all intermediate points (on the arc) and all nodes will be (m,j)-almost-complementary
#   check if a complementary node, if so break and add the next player
#   if not a complementary node, ensure it is not a initial node of the N-1 player game (some player plays move with probability 1), if it is use it to subvert to the N-1 game
#   recalculate n, i, and continue strategy
#   
# QUESTIONS
# do you only vary x[n][i] positive? How does any x[n][i] become 0 then?
###
def calculate_nash_equilibrium(N, S, U):
    # Nash equilibrium for an N-person game following the strategy of Robert Wilson
    print("Calculating Nash Equilibrium...")
    # Fix all strategies randomly for n >= 1
    x = [[1, .000, .000], [.000, .000, .000], [.000, .000, .000]]# fix_all_strategies(S)
    print("initial randomized strategies", x)
    # Find optimal strategy for player 0
    initial_node = [best_mixed_response(S, U[0], x, 0), x[1:]] # How do we ensure that best_mixed_response returns a strategy that satisfies len(S[0]) boundary conditions?
    print("initial node", initial_node)

    boundary_conditions, num_conditions_satisfied, complementary = calculate_all_boundary_conditions(U, x, N)
    print("boundary conditions", boundary_conditions)
    print("num conditions satisfied", num_conditions_satisfied)
    print("complementary", complementary)

    # most recently freed player, number of freed players in the game is n + 1
    k = 1

    while k < N:
        K = sum(len(S[i]) for i in range(k)) # number of strategies for the k player game
        # calculate boundary conditions for the new node
        # if the boundary conditions are complementary, increase k
        # if the boundary conditions are not complementary, find the n, i, s.t. x[n][i] = 0 and y[n][i] = 0
        # increment x[n][i] by epsilon-small intervals and check if K conditions are again satisfied (will otherwise be K-1)
        # if K conditions are satisfied, this is a node, check if complementary and if so increase k, if not repeat from finding n, i
        # if K conditions are not satisfied, continue
        break

    for i in range(N):
        print("- utility expectation for player", i, neg_expected_utility(x, U[i], i))

def two_player_utility(n):
    m = 0 if n == 1 else 1
    return lambda x: x[n][1]

def rock_paper_scissors_utility(n):
    def utility(x):
        sum = 0
        strategy = x[n]
        for i in range(len(x)):
            if i != n:
                sum += x[i][0] * (strategy[1] * -1 + strategy[2] * -2 + strategy[0] * -1.5)  # opponent plays rock
                sum += x[i][1] * (strategy[2] * -1 + strategy[0] * -2 + strategy[1] * -1.5)  # opponent plays paper
                sum += x[i][2] * (strategy[0] * -1 + strategy[1] * -2 + strategy[2] * -1.5)  # opponent plays scissors
        return sum
    return utility

if __name__ == "__main__":
    # 2 player example
    print("2 player example")
    S = [[0, 1], [0, 1]]
    U = [two_player_utility(0), two_player_utility(1)]
    assert(len(S) == len(U))
    N = len(S)
    #calculate_nash_equilibrium(N, S, U)

    # rock paper scissors example
    print("\n\nrock paper scissors example")
    S = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    U = [rock_paper_scissors_utility(0), rock_paper_scissors_utility(1), rock_paper_scissors_utility(2)]
    assert(len(S) == len(U))
    N = len(S)
    calculate_nash_equilibrium(N, S, U)
