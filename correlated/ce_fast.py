# python implementation of correlated equilibrium
from itertools import product
import numpy as np
import random
from scipy.optimize import linprog
from typing import List, Dict, Callable, Union, Tuple

class Correlated_equilibrium:
    strategy_map: List[str] = []
    strategies: List[int] = []
    player_map: List[str] = []
    players: List[int] = []
    utilities: List[Callable[[List[int]], float]] = []
    distribution: List[Dict[str, Union[float, Dict[str, str]]]] = None

    debug: bool = False

    def __init__(self, strategies: List[str], debug: bool = False):
        self.strategy_map = strategies
        self.strategies = [i for i in range(len(strategies))]
        self.debug = debug

    def get_lambdas(self) -> Dict[str, float]:
        """
        Returns:
        dict: A dictionary of lambda weights for each player to be optimized over.
        """
        lambdas = {1 for _ in self.players}
        if self.debug:
            print("\nInitial lambdas:\n", lambdas)
        return lambdas
    
    def enumerate_strategy_combinations(self) -> List[List[int]]:
        """
        Enumerates all possible combinations of strategies for each player.
        
        Returns:
        list: A list of lists, where each list represents a combination of strategies with indices being players and values their strategy.
        """
        # Check if there are players and strategies
        if not self.players or not self.strategies:
            return []

        # Create a list of strategies for each player
        player_strategies = [self.strategies for _ in self.players]
        # Use itertools.product to generate all combinations
        combinations = list(product(*player_strategies))
        if self.debug:
            print("\nAll strategy combinations:", combinations)
        return combinations
    
    def get_all_strategy_profiles(self) -> List[Dict[str, str]]:
        """
        Returns:
        list: A list of dicts, where each dict represents a combination of strategies with keys being players and values their strategy.
        """
        all_combinations = self.enumerate_strategy_combinations()
        strategy_profiles = [{self.player_map[i]: self.strategy_map[strategy] for i, strategy in enumerate(combination)} for combination in all_combinations]
        if self.debug:
            print("\nAll strategy profiles:", strategy_profiles)
        return strategy_profiles
    
    def build_ic_constraints(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds the inequality constraints for the linear program.
        
        Returns:
        tuple: A tuple containing the inequality constraint matrix (A_ub) and the inequality constraint vector (b_ub).
        """
        num_constraints = len(self.players) * len(self.strategies) * (len(self.strategies) - 1)
        num_variables = len(self.distribution)
        A_ub = np.zeros((num_constraints, num_variables))
        b_ub = np.zeros(num_constraints)
        constraint_index = 0
        for player in self.players:
            for strategy in self.strategies:
                for alternate_strategy in self.strategies:
                    # each row in the A_ub matrix corresponds to a constraint for a given player to play a given strategy vs an alternate strategy, 
                    # where there is an entry for each strategy profile. the value of any index of the row is non-zero iff in the profile you are signaled to 
                    # play the given strategy, and the value is the difference in utility between the alternate strategy and the signaled strategy
                    #
                    # If a row times the probability distribution vector is positive, it means that deviation given that strategy is a utility benefit, 
                    # so we constrain it to be less than or equal to 0
                    if strategy != alternate_strategy:
                        for index, dist_entry in enumerate(self.distribution):
                            profile = dist_entry["strategy"]
                            if profile[player] == strategy:
                                player_utility = self.utilities[player](profile)
                                deviation_utility = self.utilities[player]({**profile, player: alternate_strategy})
                                A_ub[constraint_index][index] = deviation_utility - player_utility
                        constraint_index += 1
        if self.debug:
            print("\nA_ub:\n", [",".join([str(round(x, 3)) for x in row]) + "\n" for row in A_ub])
            print("\nb_ub:\n", b_ub)
        return A_ub, b_ub
        
    
    def initialize_distribution(self):
        """
        Initializes the distribution with equal probability for each strategy combination.
        """
        all_combinations = self.enumerate_strategy_combinations()
        # initialize distribution with equal probability for each combination
        self.distribution = [{"probability": 1 / len(all_combinations), "strategy": combination} for combination in all_combinations]
        if self.debug:
            print("\nDistribuiton initialized:\n", "\n".join([str(row) for row in self.distribution]))

    def optimize_distribution(self):
        """
        Optimizes the distribution using linear programming.
        """
        # create a linear program
        if not self.distribution:
            self.initialize_distribution()
        lambdas = self.get_lambdas()
        outcome_utility_sums = []
        for dist_entry in self.distribution:
            profile = dist_entry["strategy"]
            weighted_strategy_utility = 0
            for player in self.players:
                # for each player, calculate the utility of their strategy given the frequency of the opponent's strategies
                # add to total utility weighted by player's lambda weight
                player_utility = self.utilities[player](profile)
                weighted_strategy_utility += lambdas[player] * player_utility
            if self.debug:
                print("\nWeighted strategy utility for profile", profile, ":", weighted_strategy_utility)
            outcome_utility_sums.append(weighted_strategy_utility)
        # builds the opbjective function as a maximization of weighted utility sum in expectation over the distribution
        c = np.array([-1 * utility for utility in outcome_utility_sums])
        
        # constrain all probabilities to sum to 1
        A_eq = np.array([[1 for _ in self.distribution]])
        b_eq = np.array([1])

        # build IC constraints
        A_ub, b_ub = self.build_ic_constraints()
        if self.debug:
            print("\n DIMENSIONS:\n", "A_ub:", np.shape(A_ub), "b_ub:", np.shape(b_ub), "A_eq:", np.shape(A_eq), "b_eq:", np.shape(b_eq))
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        if self.debug:
            print("\nResult of optimization:\n", "\n".join([str(round(res.x[i], 3)) + " " + str(self.distribution[i]["strategy"]) for i in range(len(res.x))]))
        self.distribution = [{"probability": res.x[i], "strategy": self.distribution[i]["strategy"]} for i in range(len(self.distribution))]
        return self.distribution

    def sample_distribution(self) -> Dict[str, str]:
        """
        Samples a strategy from the distribution.
        
        Returns:
        dict: A dictionary of {player: strategy} representing the sampled strategy.
        """
        if not self.distribution:
            self.optimize_distribution()
        sample = random.uniform(0, 1)
        cumulative_probability = 0
        for item in self.distribution:
            cumulative_probability += item["probability"]
            if sample <= cumulative_probability:
                if self.debug:
                    print("\nSampled probability:", sample)
                    print("\nSampled strategy:", item["strategy"])
                return item["strategy"]
        # in case of rounding errors
        print("WARNING: Rounding errors in sampling distribution, sum of probabilities is less than 1. Returning last strategy.")
        return self.distribution[-1]["strategy"]

    def add_player(self, player: str, utility_function: Callable[[List[int]], float]):
        """
        Adds a player to the game with their utility function. Assumes strategies are the same for each player.
        """
        self.player_map.append(player)
        self.players.append(len(self.players))
        self.utilities.append(utility_function)
        self.distribution = None

if __name__ == "__main__":
    pass