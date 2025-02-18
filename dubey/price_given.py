'''
Implementation of the dubey marget game where the prices are taken as given by each player. This simplification, while 
not realistic, allows for a simpler implementation of the game. 
'''

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

'''
A strategy consists of a bid to buy (price, quantity) and a bid to sell (price, quantity)
The strategy class represents these bids together, as well as keeps track of the executed state 
of the bids
'''
class strategy:
    '''
    p: price of good to buy
    q: quantity of good to buy
    p_tilde: price of good to sell
    q_tilde: quantity of good to sell
    '''
    def __init__(self, p, q, p_tilde, q_tilde):
        self.buy_price = np.array(p)
        self.buy_quantity = np.array(q)
        self.sell_price = np.array(p_tilde)
        self.sell_quantity = np.array(q_tilde)

        # executed state of the bids, all 0 to begin with
        self.bought_price = np.array([0 for _ in range(len(p))])
        self.bought_quantity = np.array([0 for _ in range(len(p))])
        self.sold_price = np.array([0 for _ in range(len(p_tilde))])
        self.sold_quantity = np.array([0 for _ in range(len(p_tilde))])
    
    '''
    Total cost of the strategy, i.e. the cost of the bids
    '''
    def total_cost(self):
        return np.dot(self.buy_price, self.buy_quantity) - np.dot(self.sell_price, self.sell_quantity)
    
    '''
    Cost of the portion of the bids that have been executed
    '''
    def executed_cost(self):
        return np.dot(self.bought_price, self.bought_quantity) - np.dot(self.sold_price, self.sold_quantity)

    '''
    Execute a trade for a given good, quantity, and price. If buy is true, the trade is a buy, otherwise it is a sell
    '''
    def execute_trade(self, good, quantity, price, buy):
        if buy:
            # ensure the trade is compatible with the strategy, i.e. quantity and price positive, and quantity is at most the bid amount and price is at most the bid price
            assert(quantity > 0 and price > 0 and self.bought_quantity[good] + quantity <= self.buy_quantity[good] and self.buy_price[good] >= price)
            # update the price of the bought good, weighted average of the price and quantity
            if self.bought_quantity[good] > 0:
                total_quantity = self.bought_quantity[good] + quantity
                self.bought_price[good] = price * quantity / total_quantity + self.bought_price[good] * self.bought_quantity[good] / total_quantity
            else:
                self.bought_price[good] = price
            # update the quantity of the bought good
            self.bought_quantity[good] += quantity
        else:
            # ensure the trade is compatible with the strategy, i.e. quantity and price positive, and quantity is at most the bid amount and price is at least the bid price
            assert(quantity > 0 and price > 0 and self.sold_quantity[good] + quantity <= self.sell_quantity[good] and self.sell_price[good] <= price)
            # update the price of the sold good, weighted average of the price and quantity
            if self.sold_quantity[good] > 0:
                total_quantity = self.sold_quantity[good] + quantity
                self.sold_price[good] = price * quantity / total_quantity + self.sold_price[good] * self.sold_quantity[good] / total_quantity
            else:
                self.sold_price[good] = price
            # update the quantity of the sold good
            self.sold_quantity[good] += quantity
    
    def __repr__(self):
        return f"(p={self.buy_price}, q={self.buy_quantity}, p_tilde={self.sell_price}, q_tilde={self.sell_quantity})"
    

'''
A player in the game, characterized by their money, goods, and alpha for the Cobb-Douglas utility function
'''
class DubeyPlayer:
    '''
    money: initial money
    goods: initial endowment of goods
    alpha: alpha for the Cobb-Douglas utility function
    '''
    def __init__(self, money, goods, alpha):
        self.money = money
        self.goods = np.array(goods)
        self.strategies = np.array([])
        self.alpha = alpha


    '''
    Utility of a given strategy, modeled as Cobb-Douglas utility function over two goods. Agent has no utility for money, though 
    money still factors into the valid strategy set
    '''
    def get_utility_of_strategy(self, strategy):
        total_goods = self.goods + strategy.buy_quantity - strategy.sell_quantity # total goods after trade, use if update is not being called first
        # Cobb-Douglas utility function
        return total_goods[0] ** self.alpha * total_goods[1] ** (1 - self.alpha)
    
    '''
    Utility of the current endowment, i.e. the utility with no extra trades
    '''
    def get_current_utility(self):
        return self.goods[0] ** self.alpha * self.goods[1] ** (1 - self.alpha)
    
    '''
    Check if a given strategy is valid, i.e. the strategy is compatible with the player's money, goods, and endowments
    '''
    def is_valid_strategy(self, s: strategy, endowments):
        # for now not modeling post period debt, needed for optimal equilibrium, assume agent can sell all goods at price p_tilde
        return s.total_cost() <= self.money and np.all(s.sell_quantity <= self.goods) and np.all(s.buy_quantity <= endowments)
    
    '''
    Update the player's money and goods given a strategy. Only updates the executed portion of the strategy
    '''
    def update(self, s: strategy):
        self.money -= s.executed_cost()
        self.goods += s.bought_quantity - s.sold_quantity

    '''
    Choose the strategy with the highest utility out of the player's strategies
    '''
    def choose_strategy(self):
        max_utility = -float('inf')
        best_strategy = None
        for s in self.strategies:
            utility = self.get_utility_of_strategy(s)
            if utility > max_utility:
                max_utility = utility
                best_strategy = s
        return best_strategy
    
    '''
    Update the player's strategies given the prices and endowments. Sets the player's strategies to be all valid strategies
    i.e. all strategies in the player's budget set assuming purchase and sale of all goods at the given prices and not exceeding the endowments
    '''
    def update_strategies(self, prices, endowments):
        new_strategies = []
        num_goods = len(self.goods)
        # iterate over all possible quantities of goods to buy and sell, whole numbers 0-9 for each good
        # currently using the entire inner product of the range of quantities, however if prices are given then each good should be 
        # independent of the others
        for buy_quantities in product(range(10), repeat=num_goods):
            for sell_quantities in product(range(10), repeat=num_goods):
                s = strategy(prices, buy_quantities, prices, sell_quantities)
                if self.is_valid_strategy(s, endowments):
                    new_strategies.append(s)
                # TODO: if not a valid strategy, can continue to the next rollover point of the inner product, as validity is monotone in the quantities
        self.strategies = np.array(new_strategies)

'''
Class representing the game, characterized by the number of goods, the prices, the players, and the endowments
Mechanism has the ability to add players, update strategies, recieve bids, and then service bids

Currently not working under any circumstances, as it cannot partially fill bids, only working where the market clears perfectly 
under a competitive equilibrium
'''
class DubeyGame:
    '''
    num_goods: number of goods
    prices: prices of goods
    '''
    def __init__(self, num_goods, prices):
        self.num_goods = num_goods
        self.prices = prices
        self.players = []
        self.endowments = np.array([0 for _ in range(num_goods)])
        self.active_bids = []

    '''
    Add a player to the game
    player: DubeyPlayer, player to add
    '''
    def add_player(self, player):
        self.players.append(player)
        self.endowments += player.goods

    '''
    Update the strategies of all players given the prices and endowments
    '''
    def update_strategies(self):
        for player in self.players:
            player.update_strategies(self.prices, self.endowments)
    
    '''
    Get the bids of all players
    '''
    def get_bids(self):
        self.update_strategies()
        bids = []
        for player in self.players:
            bids.append(player.choose_strategy())
        self.active_bids = bids
        return bids
    
    '''
    Clear the market, i.e. execute all bids for each good, and update the players' money and goods
    '''
    def clear_market(self):
        for good in range(self.num_goods):
            total_buy_quantity = 0
            total_sell_quantity = 0
            for bid in self.active_bids:
                total_buy_quantity += bid.buy_quantity[good]
                total_sell_quantity += bid.sell_quantity[good]
            if total_buy_quantity == total_sell_quantity: # market clears, execute all bids in totality for the good
                for bid in self.active_bids:
                    if bid.buy_quantity[good] > 0:
                        bid.execute_trade(good, bid.buy_quantity[good], self.prices[good], True)
                    if bid.sell_quantity[good] > 0:
                        bid.execute_trade(good, bid.sell_quantity[good], self.prices[good], False)
            else:
                print(f"ERROR: Market not cleared for good {good}, not equilibrium!")
        # update the players' money and goods
        for i in range(len(self.players)):
            self.players[i].update(self.active_bids[i])
        self.active_bids = []

    '''
    Run the game for a given number of periods, updating strategies, getting bids, and clearing the market each period
    '''
    def run_game(self, num_periods):
        for _ in range(num_periods):
            self.update_strategies()
            self.get_bids()
            self.clear_market()


if __name__ == "__main__":
    # WORKING: 2 goods, 3 players, symmetric case
    # game = DubeyGame(2, [1, 1])
    # game.add_player(DubeyPlayer(0, [2, 2], 0.25))
    # game.add_player(DubeyPlayer(0, [2, 2], 0.5))
    # game.add_player(DubeyPlayer(0, [2, 2], 0.75))

    # WORKING: 2 goods, 3 players, asymmetric case
    game = DubeyGame(2, [1, 3])
    game.add_player(DubeyPlayer(0, [3, 1], 0.5))
    game.add_player(DubeyPlayer(0, [0, 3], 0.66))
    game.add_player(DubeyPlayer(0, [9, 0], 0.33))
    game.update_strategies()
    bids = game.get_bids()
    for bid in bids:
        print(f"Purchase Quantity: {bid.buy_quantity}, Sell Quantity: {bid.sell_quantity}")
    game.clear_market()
    for player in game.players:
        print(f"Player {player.money}, {player.goods}")

'''
Currently working for symmetric case
'''

