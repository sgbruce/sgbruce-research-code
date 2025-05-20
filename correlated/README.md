# Correlated Equilibrium Directory

This directory contains Python implementations and tests for computing correlated equilibria in various game-theoretic scenarios. The main components of this directory are the implementations of the correlated equilibrium algorithms and the corresponding test cases. All equilibrium computations are done using linear programming.

## Files

### 1. `ce_basic.py`
- **Description**: This file contains a basic implementation of the correlated equilibrium algorithm. It uses linear programming to find the optimal strategy distribution that maximizes total welfare while ensuring no player has an incentive to deviate.
- **Key Classes/Functions**:
  - `Correlated_equilibrium`: A class that represents the correlated equilibrium framework, allowing for the addition of players and strategies, and computation of the equilibrium distribution.

### 2. `ce_fast.py`
- **Description**: This file provides an optimized version of the correlated equilibrium algorithm. It aims to improve computational efficiency while maintaining the core functionality of the basic implementation. Main optimizations are found in the building of incentive constraint matrices.
- **Key Classes/Functions**:
  - `Correlated_equilibrium`: Similar to the basic version but with optimizations for faster computation.

### 3. `tests.py`
- **Description**: This file contains test cases for validating the correctness of the correlated equilibrium algorithms. It includes tests for various game scenarios such as strategy enumeration, dominant strategy examples, and more.
- **Key Functions**:
  - `test_strategy_enumeration`: Tests the enumeration of all possible strategy profiles.
  - `dominant_strategy_example`: Validates the algorithm against a dominant strategy scenario.
  - `prof_bryce_example`: Tests the algorithm with a specific game example provided by Prof Bryce.

## Running Tests

To run the tests, execute the `tests.py` file. Ensure that all dependencies are installed and accessible in your Python environment. The tests will output results to the console, indicating whether each test case has passed or failed.
