import numpy as np
from dubey_regret import Strategy
import random

class NoRegretLearner:
    '''
    A class to represent a learner in a no regret algorithm.

    Each learner has a set of strategies, a set of weights, and a set of probabilities.
    The weights are used to calculate the probabilities of each strategy.
    The probabilities are used to sample a strategy from the set of strategies.
    The all_probabilities is a matrix of all the probabilities of each strategy for each time step.
    '''
    def __init__(self, strategies, initial_weights=None):
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.weights = np.array(initial_weights) if initial_weights is not None else np.ones(self.num_strategies)
        self.probabilities = self.weights / self.weights.sum()
        self.all_probabilities = np.array([self.probabilities])
        self.all_choices = np.array([])

    '''
    Sample a strategy from the set of strategies based on the current probabilities.
    '''
    def sample_strategy(self):
        choice = np.random.choice(self.strategies, p=self.probabilities)
        self.all_choices = np.append(self.all_choices, choice)
        return choice

    '''
    Update the weights of the strategies based on the cost vector.
    '''
    def update(self, cost_vector, eta=0.04):
        # Update weights using the multiplicative weights algorithm
        self.weights = self.weights * (1-eta) ** cost_vector
        if np.any(self.weights < 1e-100):
            sum = self.weights.sum()
            self.weights = self.weights / sum
        self.probabilities = self.weights / self.weights.sum()
        self.all_probabilities = np.vstack((self.all_probabilities, self.probabilities))

class DubeyLearner(NoRegretLearner):
    def __init__(self, strategies, money, goods, initial_weights=None):
        super().__init__(strategies, initial_weights)
        self.money = money
        self.goods = np.array(goods)

    def is_valid_strategy(self, s: Strategy):
        # for now not modeling post period debt, needed for optimal equilibrium 
        return np.all(s.sell_quantity <= self.goods)
    
    def update_strategies(self):
        pass

    def sample_strategy(self):
        masked_probabilities = self.probabilities * [int(self.is_valid_strategy(self.strategies[i])) for i in range(self.num_strategies)]
        total_prob = masked_probabilities.sum()
        if total_prob == 0:
            return random.choice(self.strategies)
        return np.random.choice(self.strategies, p=masked_probabilities / total_prob)
    
    # def update(self, cost_vector, eta=0.1):
    #     super().update(cost_vector, eta)
