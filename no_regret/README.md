# No Regret Directory

This directory contains various Python scripts and modules related to the implementation and testing of no-regret learning algorithms, particularly in the context of trading and strategy games. Below is a description of each file and its purpose:

## Files

### `dubey_regret.py`
This file contains the core implementation of the DubeyGame, which models a trading environment where players can submit bids for goods. It includes functions to compute trade amounts and execute trades based on player bids using TensorFlow for efficient computation.

### `dubey_tests.py`
This script provides tests for the `DubeyGame` to ensure that the trade function is computed correctly. It also tests the supervised learning oracle to verify that optimal strategies are computed accurately. The tests involve various strategy scenarios and loss function evaluations.

### `learner.py`
This file implements a simple no-regret learner using the multiplicative weights algorithm. It defines the `NoRegretLearner` class, which represents a learner with strategies, weights, and probabilities. The `DubeyLearner` class extends this to include specific trading constraints.

### `learner_tests.py`
This script contains tests for the no-regret learners, including examples of dominant strategy games. It tests the learners' ability to converge to optimal strategies under fixed and expected cost scenarios, using different cost matrices.

### `datagen.py`
This file is responsible for generating labeled data for supervised learning approaches. It computes optimal trading strategies for given endowments and preferences using nonlinear optimization techniques. The data generated is used to train models to predict optimal strategies.

### `jpsro.py`
This file contains the implementation of the Joint Policy Space Response Orcale (JPSRO). It uses the `DubeyGame` class create a market environment, and then implements the best response etimator, meta-solver, and meta-game used in the JPSRO algorithm.

### `jpsro_tests.py`
This script tests the environment for the Joint Policy Space Response Orcale (JPSRO). It includes tests for the trading environment with set bids and checks the correctness of trading operations. It also tests the CCE meta-solver with specific strategy examples.

### `jpsro_untruthful.py`
This is a slight variation on the JPRSO algorithm that allows for testing the JPSRO algorithm with untruthful best responses. It is a wrapper on the `jpsro.py` file that allows for switching between truthful and untruthful best responses cobb douglass exponents. 

## Additional Information

- The directory leverages TensorFlow and NumPy for numerical computations and machine learning tasks.
- The `GEKKO` optimization suite is used in `datagen.py` for solving nonlinear optimization problems.
