import numpy as np
import random
from itertools import product
import tensorflow as tf
'''
E consists of a set of traders N = { 1, . .. , n}, where each i in N is character-
ized by an initial endowment, ai in R^k and a utility function u': R^k -> R. We
assume (i) u' is continuous, concave, non-decreasing, and strictly increasing in
at least one variable, (ii) for any commodityj, there exist at least two traders who
are positively endowed withj, and at least two who "sufficiently desire" j
'''

class Strategy:
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
            assert(quantity > 0 and price > 0 and self.sold_quantity[good] + quantity <= self.sell_quantity[good] and self.sell_price[good] <= price)
            if self.sold_quantity[good] > 0:
                total_quantity = self.sold_quantity[good] + quantity
                self.sold_price[good] = price * quantity / total_quantity + self.sold_price[good] * self.sold_quantity[good] / total_quantity
            else:
                self.sold_price[good] = price
            self.sold_quantity[good] += quantity
    
    def get_executed_bid(self):
        return Strategy(self.bought_price, self.bought_quantity, self.sold_price, self.sold_quantity)

    def as_flat_array(self):
        return np.concatenate([self.buy_price, self.buy_quantity, self.sell_price, self.sell_quantity])
    
    def __repr__(self):
        return f"(p={self.buy_price}, q={self.buy_quantity}, p_tilde={self.sell_price}, q_tilde={self.sell_quantity})"
    
    def __eq__(self, other):
        if not isinstance(other, Strategy):
            return False
        return (self.buy_price.all() == other.buy_price.all() and
                self.buy_quantity.all() == other.buy_quantity.all() and
                self.sell_price.all() == other.sell_price.all() and
                self.sell_quantity.all() == other.sell_quantity.all())

    def __hash__(self):
        return hash((tuple(self.buy_price), tuple(self.buy_quantity), tuple(self.sell_price), tuple(self.sell_quantity)))


class Tensor_Strategy:
    def __init__(self, p, q, p_tilde, q_tilde):
        self.buy_price = tf.convert_to_tensor(p, dtype=tf.float32)
        self.buy_quantity = tf.convert_to_tensor(q, dtype=tf.float32)
        self.sell_price = tf.convert_to_tensor(p_tilde, dtype=tf.float32)
        self.sell_quantity = tf.convert_to_tensor(q_tilde, dtype=tf.float32)

        self.bought_price = tf.convert_to_tensor([0 for _ in range(p.shape[0])], dtype=tf.float32)
        self.bought_quantity = tf.convert_to_tensor([0 for _ in range(p.shape[0])], dtype=tf.float32)
        self.sold_price = tf.convert_to_tensor([0 for _ in range(p_tilde.shape[0])], dtype=tf.float32)
        self.sold_quantity = tf.convert_to_tensor([0 for _ in range(p_tilde.shape[0])], dtype=tf.float32)
    
    def total_cost(self):
        return tf.reduce_sum(self.buy_price * self.buy_quantity) - tf.reduce_sum(self.sell_price * self.sell_quantity)
    
    def executed_cost(self):
        return tf.reduce_sum(self.bought_price * self.bought_quantity) - tf.reduce_sum(self.sold_price * self.sold_quantity)

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
            assert(quantity > 0 and price > 0 and self.sold_quantity[good] + quantity <= self.sell_quantity[good] and self.sell_price[good] <= price)
            if self.sold_quantity[good] > 0:
                total_quantity = self.sold_quantity[good] + quantity
                self.sold_price[good] = price * quantity / total_quantity + self.sold_price[good] * self.sold_quantity[good] / total_quantity
            else:
                self.sold_price[good] = price
            self.sold_quantity[good] += quantity
    
    def get_executed_bid(self):
        return Strategy(self.bought_price, self.bought_quantity, self.sold_price, self.sold_quantity)

    def as_flat_array(self):
        return tf.concat([self.buy_price, self.buy_quantity, self.sell_price, self.sell_quantity], axis=0)
    
    def __repr__(self):
        return f"(p={self.buy_price}, q={self.buy_quantity}, p_tilde={self.sell_price}, q_tilde={self.sell_quantity})"
    
    def __eq__(self, other):
        if not isinstance(other, Strategy):
            return False
        return (self.buy_price.all() == other.buy_price.all() and
                self.buy_quantity.all() == other.buy_quantity.all() and
                self.sell_price.all() == other.sell_price.all() and
                self.sell_quantity.all() == other.sell_quantity.all())

    def __hash__(self):
        return hash((tuple(self.buy_price), tuple(self.buy_quantity), tuple(self.sell_price), tuple(self.sell_quantity)))

    

class DubeyPlayer:
    def __init__(self, money, goods):
        self.money = money
        self.goods = np.array(goods)
        self.strategies = np.array([])

    def get_utility(self, strategy):
        return 0
    
    def is_valid_strategy(self, s: Strategy):
        # for now not modeling post period debt, needed for optimal equilibrium 
        return s.total_cost() <= self.money and np.all(s.sell_quantity <= self.goods)
    
    def update(self, s: Strategy):
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
                        s = Strategy(buy_prices, buy_quantities, sell_prices, sell_quantities)
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
        for i, player in enumerate(self.players):
            bids.append([i, player.sample_strategy()])
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
    
    @tf.function
    def compute_trade_amounts(self, bids):
        num_players, _, num_goods = bids.shape
        output_bids = tf.zeros((num_players, 4, num_goods), dtype=tf.float32)

        def execute_trade(index, good, quantity, price, buy):
            if buy:
                new_quantity = output_bids[index, 1, good] + quantity
                new_price = (output_bids[index, 0, good] * output_bids[index, 1, good] + price * quantity) / new_quantity
                new_bids = tf.tensor_scatter_nd_update(output_bids, [[index, 1, good]], [new_quantity])
                new_bids = tf.tensor_scatter_nd_update(new_bids, [[index, 0, good]], [new_price])
            else:
                new_quantity = output_bids[index, 3, good] + quantity
                new_price = (output_bids[index, 2, good] * output_bids[index, 3, good] + price * quantity) / new_quantity
                new_bids = tf.tensor_scatter_nd_update(output_bids, [[index, 3, good]], [new_quantity])
                new_bids = tf.tensor_scatter_nd_update(new_bids, [[index, 2, good]], [new_price])
            return new_bids

        for good in range(num_goods):
            # Separate buy and sell bids for the current good, index 1 is buy quantity, index 3 is sell quantity
            buy_mask = bids[:, 1, good] > 0
            sell_mask = bids[:, 3, good] > 0
            buy_bids = tf.boolean_mask(bids, buy_mask)
            sell_bids = tf.boolean_mask(bids, sell_mask)

            if buy_bids.shape[0] == 0 or sell_bids.shape[0] == 0:
                continue

            buy_prices = bids[:, 0, good]
            sell_prices = bids[:, 2, good]

            buy_ordering = tf.argsort(buy_prices, direction='DESCENDING')
            sell_ordering = tf.argsort(sell_prices)
            # Initialize indices for buyers and sellers
            buy_index = 0
            sell_index = 0
            buyer_index = buy_ordering[buy_index]
            seller_index = sell_ordering[sell_index]
            buy_price = bids[buyer_index][0][good]
            buy_quantity = bids[buyer_index][1][good]
            sell_price = bids[seller_index][2][good]
            sell_quantity = bids[seller_index][3][good]

            # Process bids
            while buy_index < buy_ordering.shape[0] and sell_index < sell_ordering.shape[0] and buy_price >= sell_price:
                # Determine the transaction quantity between the buyer and seller
                transaction_quantity = tf.minimum(buy_quantity, sell_quantity)
                if transaction_quantity > 0:
                    buy_quantity -= transaction_quantity
                    sell_quantity -= transaction_quantity

                    # Update quantities
                    output_bids = execute_trade(buyer_index, good, transaction_quantity, buy_price, True)
                    output_bids = execute_trade(seller_index, good, transaction_quantity, buy_price, False)

                # Move to the next buyer or seller if their quantity is exhausted
                if buy_quantity == 0:
                    buy_index += 1
                    if buy_index < buy_ordering.shape[0]:
                        buyer_index = buy_ordering[buy_index]
                        buy_price = bids[buyer_index][0][good]
                        buy_quantity = bids[buyer_index][1][good]
                if sell_quantity == 0:
                    sell_index += 1
                    if sell_index < sell_ordering.shape[0]:
                        seller_index = sell_ordering[sell_index]
                        sell_price = bids[seller_index][2][good]
                        sell_quantity = bids[seller_index][3][good]

                # Implement proportional rationing if needed
                # (This part can be expanded based on specific rules for rationing)
        return output_bids
    
    def compute_trade_amounts_bilateral(self, bid, other_bid):
        executed_trade = Strategy(bid.buy_price.copy(), bid.buy_quantity.copy(), bid.sell_price.copy(), bid.sell_quantity.copy())
        for good in range(self.num_goods):
            if bid.buy_quantity[good] > 0 and other_bid.sell_quantity[good] > 0:
                transaction_quantity = min(bid.buy_quantity[good], other_bid.sell_quantity[good])
                executed_trade.execute_trade(good, transaction_quantity, bid.buy_price[good], True)
            if bid.sell_quantity[good] > 0 and other_bid.buy_quantity[good] > 0:
                transaction_quantity = min(bid.sell_quantity[good], other_bid.buy_quantity[good])
                executed_trade.execute_trade(good, transaction_quantity, other_bid.buy_price[good], False)
        return executed_trade
    
    def get_utility(self, player_index, executed_bid):
        player = self.players[player_index]
        net_money = player.money - executed_bid.executed_cost()
        if player_index == 0:
            return (player.goods[0] + executed_bid.bought_quantity[0]) + 0.5 * min(net_money, 0)
        else:
            return (player.goods[1] + executed_bid.bought_quantity[1]) + 0.5 * min(net_money, 0)
    
    def get_cost_vector(self, player_index, other_bids, debug=False):
        cost_vector = np.array([0 for _ in range(len(self.players[player_index].strategies))])
        for i, strategy in enumerate(self.players[player_index].strategies):
            executed_bid = self.compute_trade_amounts_bilateral(strategy, other_bids[0]) # 2 player case
            cost_vector[i] = - self.get_utility(player_index, executed_bid)
        # Normalize the cost vector to [0,1]
        min_cost = np.min(cost_vector)
        max_cost = np.max(cost_vector)
        if max_cost > min_cost:
            cost_vector = (cost_vector - min_cost) / (max_cost - min_cost)
            cost_vector = np.array([1 if not self.players[player_index].is_valid_strategy(self.players[player_index].strategies[i]) else cost_vector[i] for i in range(len(cost_vector))])
            if debug:
                print("COST VECTOR for player ", player_index)
                print(other_bids)
                print(cost_vector)
        else:
            cost_vector = np.zeros_like(cost_vector)
        return cost_vector
    
    def run_mechanism(self, debug=False):
        self.get_bids()
        for i, player in enumerate(self.players):
            other_bids = [bid[1] for bid in self.active_bids if bid[0] != i]
            cost_vector = self.get_cost_vector(i, other_bids, debug)
            player.update(cost_vector)


def get_all_strategies(num_goods, restricted=False):
    strategies = []  # List to maintain order
    seen = set()     # Set to track unique strategies

    max_nums = [2,2] if restricted else [6,6]

    for buy_prices in product(range(1,max_nums[0]), repeat=num_goods):
        for buy_quantities in product(range(max_nums[1]), repeat=num_goods):
            reduced_buy_prices = list(buy_prices)
            # Skip redundant buy strategies where buy quantity is zero
            for i in range(num_goods):
                if buy_quantities[i] == 0:
                    reduced_buy_prices[i] = 0

            for sell_prices in product(range(1,max_nums[0]), repeat=num_goods):
                for sell_quantities in product(range(max_nums[1]), repeat=num_goods):
                    reduced_sell_prices = list(sell_prices)
                    # Skip redundant sell strategies where sell quantity is zero
                    for i in range(num_goods):
                        if sell_quantities[i] == 0:
                            reduced_sell_prices[i] = 0

                    s = Strategy(reduced_buy_prices, buy_quantities, reduced_sell_prices, sell_quantities)
                    
                    if s not in seen:
                        seen.add(s)
                        strategies.append(s)
    return strategies


if __name__ == "__main__":
    game = DubeyGame(2)
    game.add_player(DubeyPlayer(10, [1, 1]))
    game.add_player(DubeyPlayer(10, [1, 1]))
    # print(game.get_bids())
    bids = []
    bids.append(Strategy([1,1], [4,0], [1,1], [0,1]))
    bids.append(Strategy([1,1], [0,1], [1,1], [1,0]))
    bids.append(Strategy([1,2], [0,1], [2,1], [2,0]))
    game.compute_trade_amounts(bids)
    print(f"BID 1: {bids[0].bought_quantity}, {bids[0].bought_price}, {bids[0].sold_quantity}, {bids[0].sold_price}")
    print(f"BID 2: {bids[1].bought_quantity}, {bids[1].bought_price}, {bids[1].sold_quantity}, {bids[1].sold_price}")
    print(f"BID 3: {bids[2].bought_quantity}, {bids[2].bought_price}, {bids[2].sold_quantity}, {bids[2].sold_price}")
    # print(len(get_all_strategies(2)))

