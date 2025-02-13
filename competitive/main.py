'''
This file provides a main function to solve for the competitive equilibrium of a market.
Currently working for a market with 2 goods and n agents (with Cobb-Douglas utility functions).
'''

import numpy as np
from scipy.optimize import fsolve
from competitive_market import CompetitiveMarket

def competitive_equilibrium(m):
    # Row sums
    sums = np.sum(m.endowments, axis=0)

    # System of equations
    def equations(vars):
        y = vars[:-1]
        p_b = vars[-1]

        # relationship between good a and good b in a Cobb-Douglas utility function
        # computed from MRS = p_b / p_a where p_a is taken to be 1
        # alpha is the agent's alpha in the Cobb-Douglas utility function
        def get_x_from_y(y, alpha):
            return  y * p_b * alpha / (1-alpha)

        # Budget constraints for each agent, given prices cannot exceed endowment with trade
        budgets = [get_x_from_y(y[i], m.alphas[i]) + p_b * y[i] - (m.endowments[i, 0] + p_b * m.endowments[i, 1]) for i in range(len(y))]

        # Additional constraints
        a_clearing = np.sum([get_x_from_y(y[i], m.alphas[i]) for i in range(len(y))]) - sums[0]
        b_clearing = np.sum(y) - sums[1]

        # Return the equations as a list. the system is overdetermined, so we ignore the last clearing equation
        return [*budgets, a_clearing]

    # Initial guess for (x1, x2, p1)
    intial_budget_guess = [1 for _ in range(len(m.endowments))]
    initial_guess = [*intial_budget_guess, 1]

    # Solve the system
    solution = fsolve(equations, initial_guess)

    y = solution[:-1]
    p_b = solution[-1]
    def get_x_from_y(y, alpha):
        return y * p_b * alpha / (1 - alpha)
    
    x = [get_x_from_y(y[i], m.alphas[i]) for i in range(len(y))]

    # Concatenate the full outcomes for each agent in terms of goods x and y
    full_outcomes = np.array([x, y])
    m.update_market(full_outcomes, [1, p_b])

if __name__ == "__main__":
    # Given W matrix
    W = np.array([[2,2],[2,2],[2,2]])
    a = [0.25, 0.5, 0.75]
    market = CompetitiveMarket(W, a)

    competitive_equilibrium(market)
    market.print_market()