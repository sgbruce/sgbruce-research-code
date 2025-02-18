import numpy as np
from competitive_market import CompetitiveMarket as cm
from ce_from_market import Correlated_equilibrium as ce

def get_market_constraints(market: cm):
    # how to represent the busget constraint as a function of 
    # strategies? (and probabilities)
    #
    # ANSWER: restriction on the strategy space, see notes
    # 
    pass

def cd_utility(player, alpha):
    p = int(player)
    def utility(profile, strategy_map):
        allocation = [int(a) for a in strategy_map[profile[p]].split(',')]
        return allocation[0] ** alpha * allocation[1] ** (1 - alpha)
    return utility

def game_from_market(market: cm):
    strategies = ['1,1', '1,2', '2,1', '2,2', '3,1', '3,2', '1,3', '2,3', '3,3']
    c = ce(strategies, debug=True)
    c.add_player("0", cd_utility("0", market.alphas[0]))
    c.add_player("1", cd_utility("1", market.alphas[1]))
    c.add_constraints(get_market_constraints(market))
    c.initialize_distribution()
    distribution = c.optimize_distribution()
    print(distribution)

    return distribution

c = cm([[1, 2], [3, 4]], [0.5, 0.5])
game_from_market(c)