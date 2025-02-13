'''
This code attempts to find the competitive equilibrium of a market with multiple agents. Each agent has a Cobb-Douglas utility function.
It attempts to compute the competitive equilibrium by solving the system of equations that arise from the first order conditions of the utility maximization problem.
Currently not working
'''

import numpy as np
from scipy.optimize import fsolve

'''
W: matrix of endowments
A: array of alphas
'''
def competitive_equilibrium(W, A):
    # Row sums
    sums = np.sum(W, axis=0)

    # System of equations
    def equations(vars):
        x = vars[:-1]
        p2 = vars[-1]

        p = np.array([1, p2])

        def get_coeff(a):
            return  a / (1-a) * p2

        # Equations from pO = pW
        coeffs = np.array([get_coeff(a) for a in A])
        eqs = p @ np.array([coeffs * x, x]) - p @ W
        print(p @ np.array([coeffs * x, x]) - p @ W, "\n\n\n")

        # Additional constraints
        eq4 = np.sum(x) - sums[1]
        eq5 = sum(get_coeff(A[i]) * x[i] for i in range(len(A))) - sums[1]

        return np.concatenate((eqs, [eq4]))

    # Initial guess for (x1, x2, p1)
    initial_guess = [1 for _ in range(len(A))] + [1]

    # Solve the system
    solution = fsolve(equations, initial_guess)

    x = solution[:-1]
    p2 = solution[-1]
    return x, p2

if __name__ == "__main__":
    a1 = 0.25
    a2 = 0.5
    a3 = 0.75
    # Given W matrix
    W = np.array([[2, 2, 2], [2, 2, 2]])
    print(competitive_equilibrium(W, [a1, a2, a3]))
