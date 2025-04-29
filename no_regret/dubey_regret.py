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
        """
        Computes the net buy/sell quantities and average prices for each player and good.

        Args:
            bids: A tensor of shape (num_players, 4, num_goods), where:
                - bids[:, 0, g] = buy price for good g
                - bids[:, 1, g] = buy quantity for good g
                - bids[:, 2, g] = sell price for good g
                - bids[:, 3, g] = sell quantity for good g

        Returns:
            output_bids: A tensor of shape (num_players, 4, num_goods) with:
                - output[:, 0, g] = average buy price for good g
                - output[:, 1, g] = total bought quantity
                - output[:, 2, g] = average sell price
                - output[:, 3, g] = total sold quantity
        """
        num_players = tf.shape(bids)[0]
        num_goods = tf.shape(bids)[2]
        output_bids = tf.zeros((num_players, 4, num_goods), dtype=tf.float32)
        eps = tf.constant(1e-8, dtype=tf.float32)

        def weighted_avg_price(old_price, old_qty, new_price, new_qty):
            total_qty = old_qty + new_qty
            safe_qty = tf.maximum(total_qty, eps)
            return (old_price * old_qty + new_price * new_qty) / safe_qty

        def execute_trade(output, index, good, qty, price, is_buy):
            price_idx = 0 if is_buy else 2
            qty_idx = 1 if is_buy else 3

            old_qty = output[index, qty_idx, good]
            old_price = output[index, price_idx, good]

            new_qty = old_qty + qty
            avg_price = weighted_avg_price(old_price, old_qty, price, qty)

            output = tf.tensor_scatter_nd_update(output, [[index, qty_idx, good]], [new_qty])
            output = tf.tensor_scatter_nd_update(output, [[index, price_idx, good]], [avg_price])
            return output

        for good in tf.range(num_goods):
            buy_prices = bids[:, 0, good]
            sell_prices = bids[:, 2, good]

            buy_order = tf.argsort(buy_prices, direction="DESCENDING")
            sell_order = tf.argsort(sell_prices)

            buy_index = tf.constant(0)
            sell_index = tf.constant(0)

            buyer_idx = buy_order[buy_index]
            seller_idx = sell_order[sell_index]

            buy_price = tf.gather(bids[:, 0, good], buyer_idx)
            buy_qty = tf.gather(bids[:, 1, good], buyer_idx)
            sell_price = tf.gather(bids[:, 2, good], seller_idx)
            sell_qty = tf.gather(bids[:, 3, good], seller_idx)

            def loop_cond(output, bi, si, b_idx, s_idx, bp, bq, sp, sq):
                return tf.logical_and(
                    tf.logical_and(bi < num_players, si < num_players),
                    bp >= sp
                )

            def loop_body(output, bi, si, b_idx, s_idx, bp, bq, sp, sq):
                trade_qty = tf.minimum(bq, sq)
                should_trade = trade_qty > 0

                def trade():
                    out = execute_trade(output, b_idx, good, trade_qty, bp, True)
                    out = execute_trade(out, s_idx, good, trade_qty, bp, False)
                    return out

                output = tf.cond(should_trade, trade, lambda: output)

                bq = tf.cond(should_trade, lambda: bq - trade_qty, lambda: bq)
                sq = tf.cond(should_trade, lambda: sq - trade_qty, lambda: sq)

                next_bi = bi + 1
                has_next_buyer = next_bi < num_players
                next_buyer_idx = tf.cond(has_next_buyer, lambda: buy_order[next_bi], lambda: b_idx)
                advance_buyer = tf.logical_and(tf.equal(bq, 0), has_next_buyer)

                bi = tf.cond(tf.equal(bq, 0), lambda: next_bi, lambda: bi)
                b_idx = tf.cond(advance_buyer, lambda: next_buyer_idx, lambda: b_idx)
                bp = tf.cond(advance_buyer, lambda: tf.gather(bids[:, 0, good], next_buyer_idx), lambda: bp)
                bq = tf.cond(advance_buyer, lambda: tf.gather(bids[:, 1, good], next_buyer_idx), lambda: bq)

                next_si = si + 1
                has_next_seller = next_si < num_players
                next_seller_idx = tf.cond(has_next_seller, lambda: sell_order[next_si], lambda: s_idx)
                advance_seller = tf.logical_and(tf.equal(sq, 0), has_next_seller)

                si = tf.cond(tf.equal(sq, 0), lambda: next_si, lambda: si)
                s_idx = tf.cond(advance_seller, lambda: next_seller_idx, lambda: s_idx)
                sp = tf.cond(advance_seller, lambda: tf.gather(bids[:, 2, good], next_seller_idx), lambda: sp)
                sq = tf.cond(advance_seller, lambda: tf.gather(bids[:, 3, good], next_seller_idx), lambda: sq)

                return output, bi, si, b_idx, s_idx, bp, bq, sp, sq

            loop_vars = [output_bids, buy_index, sell_index, buyer_idx, seller_idx, buy_price, buy_qty, sell_price, sell_qty]
            output_bids, *_ = tf.while_loop(loop_cond, loop_body, loop_vars, maximum_iterations=num_players * 2 + 1)

        return output_bids
    
    @tf.function
    def compute_trade_amounts_vectorized(self, bids: tf.Tensor) -> tf.Tensor:
        """
        Vectorized batched implementation of trade amount computation.
        
        Args:
            bids: Tensor of shape (batch_size, num_players, 4, num_goods)
                with channels: [buy_price, buy_quantity, sell_price, sell_quantity]
        
        Returns:
            Tensor of shape (batch_size, num_players, 4, num_goods) with:
            [avg_buy_price, total_buy_quantity, avg_sell_price, total_sell_quantity]
        """
        batch_size = tf.shape(bids)[0]
        num_players = tf.shape(bids)[1]
        num_goods = tf.shape(bids)[3]

        # Unpack bid components
        buy_price = bids[:, :, 0, :]  # shape (B, P, G)
        buy_qty = bids[:, :, 1, :]
        sell_price = bids[:, :, 2, :]
        sell_qty = bids[:, :, 3, :]

        # Prepare sorting indices
        buy_sort_indices = tf.argsort(buy_price, axis=1, direction='DESCENDING')
        sell_sort_indices = tf.argsort(sell_price, axis=1)

        # Compute inverse indices to restore original player order
        buy_inverse_indices = tf.argsort(buy_sort_indices, axis=1)
        sell_inverse_indices = tf.argsort(sell_sort_indices, axis=1)

        # Sort prices and quantities accordingly (batch gather)
        def batch_gather(tensor, indices):
            batch_range = tf.range(tf.shape(tensor)[0])[:, tf.newaxis, tf.newaxis]  # shape (B, 1, 1)
            good_range = tf.range(tf.shape(tensor)[2])[tf.newaxis, tf.newaxis, :]   # shape (1, 1, G)
            batch_indices = tf.tile(batch_range, [1, tf.shape(tensor)[1], tf.shape(tensor)[2]])
            good_indices = tf.tile(good_range, [tf.shape(tensor)[0], tf.shape(tensor)[1], 1])
            gather_indices = tf.stack([batch_indices, indices, good_indices], axis=-1)
            return tf.gather_nd(tensor, gather_indices)

        # of shape (B, P, G), where [0,0,0] is the highest buy price for good 0, batch 0, [0,1,0] is the second highest buy price for good 0, batch 0, [0,0,1] is the highest buy price for good 1, batch 0, etc.
        # same for all prices and quantities
        sorted_buy_price = batch_gather(buy_price, buy_sort_indices)
        sorted_buy_qty = batch_gather(buy_qty, buy_sort_indices)
        sorted_sell_price = batch_gather(sell_price, sell_sort_indices)
        sorted_sell_qty = batch_gather(sell_qty, sell_sort_indices)
        # Initialize output tensor
        output = tf.zeros_like(bids)

        def compute_for_good(good_idx):
            # shape: (B, P), where [0,0] is the highest buy price for batch 0, [0,1] is the second highest buy price for batch 0, [1,0] is the highest buy price for batch 1, etc.
            # same for all prices quantities
            bp = sorted_buy_price[:, :, good_idx]
            bq = sorted_buy_qty[:, :, good_idx]
            sp = sorted_sell_price[:, :, good_idx]
            sq = sorted_sell_qty[:, :, good_idx]

            # Match buyers and sellers greedily
            buyer_ptr = tf.zeros([batch_size], dtype=tf.int32)
            seller_ptr = tf.zeros([batch_size], dtype=tf.int32)

            # Buffers for accumulating results
            buy_sum_price = tf.zeros([batch_size, num_players])
            buy_sum_qty = tf.zeros([batch_size, num_players])
            sell_sum_price = tf.zeros([batch_size, num_players])
            sell_sum_qty = tf.zeros([batch_size, num_players])

            mask = tf.ones((batch_size,), dtype=tf.bool)

            cond = lambda bp, bq, sp, sq, buyer_ptr, seller_ptr, buy_sum_price, buy_sum_qty, sell_sum_price, sell_sum_qty, mask: tf.reduce_any(mask)

            def body(bp, bq, sp, sq, buyer_ptr, seller_ptr,
                        buy_sum_price, buy_sum_qty, sell_sum_price, sell_sum_qty, mask):

                # 1D of length B (batch_size), where each element is the sorted index of the player that is buying/selling for that batch index, NOT PLAYER INDEX, ALWAYS STARTING FROM 0
                b_idx = buyer_ptr
                b_idx = tf.where(mask, b_idx, tf.zeros((batch_size), dtype=tf.int32))
                s_idx = seller_ptr
                s_idx = tf.where(mask, s_idx, tf.zeros((batch_size), dtype=tf.int32))

                # Gather current bids, of shape B, where each element is the value of the buy/sell price/quantity for the player active at that bid index
                b_price = tf.where(mask, tf.gather(bp, b_idx, axis=1, batch_dims=1), tf.zeros((batch_size)))
                b_quantity = tf.where(mask, tf.gather(bq, b_idx, axis=1, batch_dims=1), tf.zeros((batch_size)))
                s_price = tf.where(mask, tf.gather(sp, s_idx, axis=1, batch_dims=1), tf.zeros((batch_size)))
                s_quantity = tf.where(mask, tf.gather(sq, s_idx, axis=1, batch_dims=1), tf.zeros((batch_size)))

                # Check trade condition
                should_trade = b_price >= s_price
                trade_qty = tf.where(should_trade, tf.minimum(b_quantity, s_quantity), tf.zeros_like(b_quantity))

                # Masks keep track of the active buyer/seller to subtract the trade quantity from
                buyer_mask = tf.where(mask[:, tf.newaxis], tf.one_hot(b_idx, num_players, dtype=tf.int32), tf.zeros((batch_size, num_players), dtype=tf.int32))
                seller_mask = tf.where(mask[:, tf.newaxis], tf.one_hot(s_idx, num_players, dtype=tf.int32), tf.zeros((batch_size, num_players), dtype=tf.int32))

                # Update the buy price/quantity for each batch index for the active player
                buy_sum_price += trade_qty[:, tf.newaxis] * b_price[:, tf.newaxis] * tf.cast(buyer_mask, dtype=tf.float32)
                buy_sum_qty += trade_qty[:, tf.newaxis] * tf.cast(buyer_mask, dtype=tf.float32)

                # Update the sell price/quantity for each batch index for the active player
                sell_sum_price += trade_qty[:, tf.newaxis] * b_price[:, tf.newaxis] * tf.cast(seller_mask, dtype=tf.float32)
                sell_sum_qty += trade_qty[:, tf.newaxis] * tf.cast(seller_mask, dtype=tf.float32)

                # Update remaining buy/sell quantities for each batch index for the active player
                b_indices = tf.expand_dims(tf.where(mask, b_idx, tf.zeros((batch_size), dtype=tf.int32)), axis=-1)
                b_indices = tf.concat([tf.range(tf.shape(b_indices)[0], dtype=tf.int32)[:, tf.newaxis], b_indices], axis=-1)
                bq = tf.tensor_scatter_nd_sub(bq, b_indices, trade_qty)
                
                s_indices = tf.expand_dims(tf.where(mask, s_idx, tf.zeros((batch_size), dtype=tf.int32)), axis=-1)
                s_indices = tf.concat([tf.range(tf.shape(s_indices)[0], dtype=tf.int32)[:, tf.newaxis], s_indices], axis=-1)
                sq = tf.tensor_scatter_nd_sub(sq, s_indices, trade_qty)
                
                # Update the buyer/seller player index to the next active player where no quantity is left
                buyer_ptr += tf.where(mask, tf.cast(tf.gather(bq, b_idx, axis=1, batch_dims=1) == 0, tf.int32), tf.zeros((batch_size), dtype=tf.int32))
                seller_ptr += tf.where(mask, tf.cast(tf.gather(sq, s_idx, axis=1, batch_dims=1) == 0, tf.int32), tf.zeros((batch_size), dtype=tf.int32))

                still_active = tf.logical_and(tf.logical_and(buyer_ptr < num_players, seller_ptr < num_players), should_trade)
                mask = tf.logical_and(mask, still_active)

                return bp, bq, sp, sq, buyer_ptr, seller_ptr, buy_sum_price, buy_sum_qty, sell_sum_price, sell_sum_qty, mask

            max_loops = num_players * 2 + 1 # buyer or seller has to change on each loop, so must finish in this many loops
            bp_fin, _, sp_fin, _, _, _, bp_sum, bq_sum, sp_sum, sq_sum, mask = tf.while_loop(
                cond, body,
                [bp, bq, sp, sq, buyer_ptr, seller_ptr, buy_sum_price, buy_sum_qty, sell_sum_price, sell_sum_qty, mask],
                shape_invariants=[
                    tf.TensorShape([None, None]),  # bp
                    tf.TensorShape([None, None]),  # bq
                    tf.TensorShape([None, None]),  # sp
                    tf.TensorShape([None, None]),  # sq
                    tf.TensorShape([None]),        # buyer_ptr
                    tf.TensorShape([None]),        # seller_ptr
                    tf.TensorShape([None, None]),  # buy_sum_price
                    tf.TensorShape([None, None]),  # buy_sum_qty
                    tf.TensorShape([None, None]),  # sell_sum_price
                    tf.TensorShape([None, None]),  # sell_sum_qty
                    tf.TensorShape([None]),        # mask
                ],
                maximum_iterations=max_loops
            )

            # Avoid div by zero
            avg_buy_price = tf.where(bq_sum > 0, bp_sum / bq_sum, bp_fin)
            avg_sell_price = tf.where(sq_sum > 0, sp_sum / sq_sum, sp_fin)

            # Restore original player order
            buy_inv_idx = buy_inverse_indices[:, :, good_idx]
            sell_inv_idx = sell_inverse_indices[:, :, good_idx]

            avg_buy_price_reordered = tf.gather(avg_buy_price, buy_inv_idx, axis=1, batch_dims=1)
            bq_sum_reordered = tf.gather(bq_sum, buy_inv_idx, axis=1, batch_dims=1)

            avg_sell_price_reordered = tf.gather(avg_sell_price, sell_inv_idx, axis=1, batch_dims=1)
            sq_sum_reordered = tf.gather(sq_sum, sell_inv_idx, axis=1, batch_dims=1)

            # Stack into (B, P, 4) with restored order
            result = tf.stack([
                avg_buy_price_reordered,
                bq_sum_reordered,
                avg_sell_price_reordered,
                sq_sum_reordered
            ], axis=2)
            # print("result for good ", good_idx, ": ", result)

            return result

        results_per_good = tf.map_fn(compute_for_good, tf.range(num_goods), fn_output_signature=tf.TensorSpec((None, None, 4), tf.float32))
        # shape now is (G, B, P, 4), so we need to transpose
        result = tf.transpose(results_per_good, perm=[1, 2, 3, 0])  # (B, P, 4, G)
        return result

    
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

