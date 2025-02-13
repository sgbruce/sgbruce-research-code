'''
This code attempts to find the competitive equilibrium of a market with multiple agents. Each agent has a Cobb-Douglas utility function.
It attempts to find the pareto weights that correspond to the competitive equilibrium, which is guaranteed to exist via the first welfare theorem.
Currently not working becuase the objective is not accurately representing the utility functions since the objective is not linear...
'''

import numpy as np
from scipy.optimize import linprog

def compute_pareto_weights(endowments, alphas):
    """
    Compute Pareto weights from endowments and Cobb-Douglas alphas.
    """
    num_agents = len(alphas)
    pareto_weights = np.array([1 / (alpha * e_x ** (alpha - 1) * e_y ** (1 - alpha))
                               for (e_x, e_y), alpha in zip(endowments, alphas)])
    pareto_weights /= pareto_weights.sum()  # Normalize
    return pareto_weights

def solve_equilibrium(endowments, alphas):
    """
    Solve for competitive equilibrium allocations and prices via linear programming.
    """
    num_agents = len(alphas)
    
    # Total endowment (market clearing constraints)
    total_x = sum(e[0] for e in endowments)
    total_y = sum(e[1] for e in endowments)
    
    # Objective: Maximize social planner’s weighted utility sum (negative for minimization)
    c = -np.concatenate([np.log(alphas) * np.ones(num_agents), np.log(1 - np.array(alphas)) * np.ones(num_agents)])
    
    # Constraints: Sum of allocations must equal total endowment
    A_eq = np.zeros((2, 2 * num_agents))
    A_eq[0, :num_agents] = 1  # Sum of x allocations = total_x
    A_eq[1, num_agents:] = 1  # Sum of y allocations = total_y
    b_eq = [total_x, total_y]
    
    # Bounds: Each agent gets non-negative allocations
    bounds = [(0, None)] * (2 * num_agents)
    
    # Solve LP
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        raise ValueError("Linear programming failed to find an optimal allocation.")
    
    allocations = result.x.reshape(2, num_agents).T  # Reshape to [(x1, y1), (x2, y2), (x3, y3)]
    
    # Compute equilibrium prices (using marginal utilities at equilibrium allocations)
    p_x = sum(alphas / allocations[:, 0])
    p_y = sum((1 - np.array(alphas)) / allocations[:, 1])
    prices = np.array([p_x, p_y]) / p_y  # Normalize so p_y = 1
    
    return allocations, prices

# Example usage
endowments = [(2, 2), (2, 2), (2, 2)]  # Endowments for 3 players (x, y)
alphas = [0.25, 0.5, 0.75]  # Cobb-Douglas preference parameters

pareto_weights = compute_pareto_weights(endowments, alphas)
allocations, prices = solve_equilibrium(endowments, alphas)

print("Pareto Weights:", pareto_weights)
print("Final Allocations:", allocations)
print("Equilibrium Prices:", prices)
