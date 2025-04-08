import numpy as np
import random
from itertools import product
'''
E consists of a set of traders N = { 1, . .. , n}, where each i in N is character-
ized by an initial endowment, ai in R^k and a utility function u': R^k -> R. We
assume (i) u' is continuous, concave, non-decreasing, and strictly increasing in
at least one variable, (ii) for any commodityj, there exist at least two traders who
are positively endowed withj, and at least two who "sufficiently desire" j
'''

class strategy:
    def __init__(self, p, q, p_tilde, q_tilde):
        self.buy_price = np.array(p)
        self.buy_quantity = np.array(q)
        self.sell_price = np.array(p_tilde)
        self.sell_quantity = np.array(q_tilde)

        self.bought_price = np.array([0 for _ in range(len(p))])
        self.bought_quantity = np.array([0 for _ in range(len(p))])
        self.sold_price = np.array([0 for _ in range(len(p_tilde))])
        self.sold_quantity = np.array([0 for _ in range(len(p_tilde))])
    
    def total_cost(self):
        return np.dot(self.buy_price, self.buy_quantity) - np.dot(self.sell_price, self.sell_quantity)
    
    def executed_cost(self):
        return np.dot(self.bought_price, self.bought_quantity) - np.dot(self.sold_price, self.sold_quantity)

    def execute_trade(self, good, quantity, price, buy):
        if buy:
            assert(quantity > 0 and price > 0 and self.bought_quantity[good] + quantity <= self.buy_quantity[good] and self.buy_price[good] >= price)
            if self.bought_quantity[good] > 0:
                total_quantity = self.bought_quantity[good] + quantity
                self.bought_price[good] = price * quantity / total_quantity + self.bought_price[good] * self.bought_quantity[good] / total_quantity
            else:
                self.bought_price[good] = price
            self.bought_quantity[good] += quantity
        else:
            self.sold_quantity[good] += quantity
            self.sold_price[good] = price
    
    def __repr__(self):
        return f"(p={self.buy_price}, q={self.buy_quantity}, p_tilde={self.sell_price}, q_tilde={self.sell_quantity})"
    

class DubeyPlayer:
    def __init__(self, money, goods):
        self.money = money
        self.goods = np.array(goods)
        self.strategies = np.array([])

    def get_utility(self, strategy):
        return 0
    
    def is_valid_strategy(self, s: strategy):
        # for now not modeling post period debt, needed for optimal equilibrium 
        return s.total_cost() <= self.money and np.all(s.sell_quantity <= self.goods)
    
    def update(self, s: strategy):
        self.money -= s.executed_cost()
        self.goods -= s.executed_sell_quantity()
        self.goods += s.executed_buy_quantity()

    def sample_strategy(self):
        return random.choice(self.strategies)
    
    def update_strategies(self):
        new_strategies = []
        num_goods = len(self.goods)
        for buy_prices in product(range(2), repeat=num_goods):
            for buy_quantities in product(range(2), repeat=num_goods):
                for sell_prices in product(range(2), repeat=num_goods):
                    for sell_quantities in product(range(2), repeat=num_goods):
                        s = strategy(buy_prices, buy_quantities, sell_prices, sell_quantities)
                        if self.is_valid_strategy(s):
                            new_strategies.append(s)
        self.strategies = np.array(new_strategies)
'''
In each period, need to update the valid set of strategies for each player given their endowments, then 
have each player submit their next strategy, and clear the market. 

Assume no coordination at first
'''
class DubeyGame:
    def __init__(self, num_goods):
        self.num_goods = num_goods
        self.players = []
        self.active_bids = []
        self.endowments = np.array([0 for _ in range(num_goods)])

    def add_player(self, player):
        self.players.append(player)
        self.endowments += player.goods

    def update_strategies(self):
        for player in self.players:
            player.update_strategies()
    
    def get_bids(self):
        self.update_strategies()
        bids = []
        for player in self.players:
            bids.append(player.sample_strategy())
        self.active_bids = bids
        return bids
    
    def clear_market(self):
        for good in range(self.num_goods):
            total_buy_quantity = 0
            total_sell_quantity = 0
            for bid in self.active_bids:
                total_buy_quantity += bid.buy_quantity[good]
                total_sell_quantity += bid.sell_quantity[good]
            if total_buy_quantity == total_sell_quantity:
                for bid in self.active_bids:
                    if bid.buy_quantity[good] > 0:
                        bid.execute_trade(good, bid.buy_quantity[good], self.prices[good], True)
                    if bid.sell_quantity[good] > 0:
                        bid.execute_trade(good, bid.sell_quantity[good], self.prices[good], False)
            else:
                print(f"ERROR: Market not cleared for good {good}, not equilibrium!")
        for i in range(len(self.players)):
            self.players[i].update(self.active_bids[i])
        self.active_bids = []

if __name__ == "__main__":
    game = DubeyGame(2)
    game.add_player(DubeyPlayer(10, [1, 1]))
    game.add_player(DubeyPlayer(10, [1, 1]))
    print(game.get_bids())

