import tensorflow as tf

def compute_trade_amounts_vectorized(bids: tf.Tensor) -> tf.Tensor:
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

            still_active = tf.logical_and(buyer_ptr < num_players, seller_ptr < num_players)
            mask = tf.logical_and(mask, still_active)

            return bp, bq, sp, sq, buyer_ptr, seller_ptr, buy_sum_price, buy_sum_qty, sell_sum_price, sell_sum_qty, mask

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
            ]
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


def get_custom_loss_vectorized(debug = False):
    num_players = 2
    num_goods = 2
    num_inputs = 2 * num_goods * num_players
    num_outputs = 4 * num_goods * num_players
    def custom_loss_vectorized(y_true, y_pred):
        # Split context and bids
        true_context = tf.reshape(y_true[:, :num_inputs], (-1, 2, num_players, num_goods))
        pred_context = tf.reshape(y_pred[:, :num_inputs], (-1, 2, num_players, num_goods))

        endowments = true_context[:, 0]
        preferences = true_context[:, 1]

        true_bids = tf.reshape(y_true[:, num_inputs:], (-1, num_players, 4, num_goods))
        pred_bids = tf.reshape(y_pred[:, num_inputs:], (-1, num_players, 4, num_goods))

        # OVERSOLD PENALTY
        sale_quantity = pred_bids[:, :, 3]  # shape (batch, num_players, num_goods)
        oversold_amount = tf.nn.relu(sale_quantity - endowments)
        total_oversold = tf.reduce_sum(oversold_amount, axis=[1, 2])  # shape (batch,)

        # TRADE AMOUNTS (batch compute)
        true_trade_amounts = compute_trade_amounts_vectorized(true_bids)  # should be batched
        pred_trade_amounts = compute_trade_amounts_vectorized(pred_bids)  # should be batched

        def compute_utility(trade_amounts):
            bought_prices = trade_amounts[:, :, 0]
            bought_quantities = trade_amounts[:, :, 1]
            sold_prices = trade_amounts[:, :, 2]
            sold_quantities = trade_amounts[:, :, 3]
            endowed_cash = tf.constant([0 for player in range(num_players)], dtype=tf.float32)
            endowed_cash = tf.reshape(endowed_cash, (1, -1))  # shape (1, num_players)
            cash = tf.reduce_sum(sold_prices * sold_quantities - bought_prices * bought_quantities, axis=2) + endowed_cash
            cash_penalty = -tf.nn.softplus(-cash)
            player_utils = tf.reduce_sum(preferences * bought_quantities, axis=2)
            return tf.reduce_sum(player_utils - 0.5 * cash_penalty, axis=1)  # shape (batch,)

        true_util = compute_utility(true_trade_amounts)
        pred_util = compute_utility(pred_trade_amounts)

        # EXCESS BID (if needed)
        # excess_bid = tf.reduce_sum(pred_bids - pred_trade_amounts, axis=[1,2,3])

        # fallback mse
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

        total_loss = (true_util - pred_util) + 10 * total_oversold + 0.1 * mse_loss
        return tf.reduce_mean(total_loss)
    return custom_loss_vectorized

def basic_scatter_test():
    # Create a matrix of size [4, 2]
    matrix = tf.Variable([[10, 20], [30, 40], [50, 60], [70, 80]], dtype=tf.int32)
    # Indices to update
    indices = tf.constant([0, 1, 0, 1], dtype=tf.int32)
    # Update indices to be the index given into each of the elements of the input tensor
    indices = tf.expand_dims(indices, axis=-1)
    indices = tf.concat([tf.range(tf.shape(indices)[0], dtype=tf.int32)[:, tf.newaxis], indices], axis=-1)
    print(indices)
    # Values to subtract
    updates = tf.constant([5, 15, 10, 20], dtype=tf.int32)
    # Perform scatter sub update
    updated_matrix = tf.tensor_scatter_nd_sub(matrix, indices, updates)
    # Print the updated matrix
    print(updated_matrix)

if __name__ == "__main__":
    # dims: (batch_size, num_players, 4, num_goods)
    context_tensor = tf.Variable([[[[0, 0], [0, 0]], [[1, 0], [0, 1]]], \
                                  [[[0, 0], [1, 1]], [[1, 0], [0, 1]]], \
                                  [[[0, 1], [1, 0]], [[1, 0], [0, 1]]], \
                                  [[[0, 0], [1, 1]], [[1, 0], [0, 1]]], \
                                  [[[1, 1], [1, 1]], [[1, 0], [0, 1]]]], dtype=tf.float32)
    
    tensor = tf.Variable([[[[1,2],[0,0],[1,1],[0,0]], [[3,4],[0,0],[1,1],[0,0]]], \
                          [[[3,1],[2,2],[2,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]], \
                          [[[1,1],[1,0],[1,1],[0,1]], [[1,1],[0,1],[1,1],[1,0]]], \
                          [[[1,1],[1,1],[1,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]], \
                          [[[1,1],[2,2],[1,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]]], dtype=tf.float32)
    
    expected_output = tf.Variable([[[[1,2],[0,0],[1,1],[0,0]], [[3,4],[0,0],[1,1],[0,0]]], \
                                   [[[3,1],[1,1],[2,1],[0,0]], [[1,1],[0,0],[3,1],[1,1]]], \
                                   [[[1,1],[1,0],[1,1],[0,1]], [[1,1],[0,1],[1,1],[1,0]]], \
                                   [[[1,1],[1,1],[1,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]], \
                                   [[[1,1],[1,1],[1,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]]], dtype=tf.float32)
    
    output = compute_trade_amounts_vectorized(tensor)
    # print("output: ", output.numpy())
    # print("expected_output: ", expected_output.numpy())
    print("computed trade amounts as expected: ", tf.reduce_all(tf.equal(output, expected_output)))

    sub_opt_tensor = tf.Variable([[[[1,2],[0,0],[1,1],[0,0]], [[3,4],[0,0],[1,1],[0,0]]], \
                                  [[[3,1],[2,2],[2,1],[0,0]], [[1,1],[0,0],[1,1],[1,1]]], \
                                  [[[1,1],[1,0],[1,1],[0,1]], [[1,1],[0,1],[1,1],[1,0]]], \
                                  [[[1,1],[1,1],[1,1],[0,0]], [[1,1],[0,0],[1,1],[0,1]]], \
                                  [[[1,1],[1,1],[1,1],[1,1]], [[1,1],[1,1],[1,1],[1,1]]]], dtype=tf.float32)
    
    batch_optimal = tf.concat([tf.reshape(context_tensor, (context_tensor.shape[0], -1)), tf.reshape(tensor, (tensor.shape[0], -1))], axis=1)
    batch_sub_optimal = tf.concat([tf.reshape(context_tensor, (context_tensor.shape[0], -1)), tf.reshape(sub_opt_tensor, (sub_opt_tensor.shape[0], -1))], axis=1)

    custom_loss = get_custom_loss_vectorized()
    output = custom_loss(batch_optimal, batch_sub_optimal)
    print("output: ", output.numpy())

    t = tf.Variable([[[[27, 48], [81, 31], [9, 15], [9, 2]], [[1.0, 0.76653906543], [0.0, 31.0], [1.0, 0.76653906543], [23.7627110284, 0.0]], [[1.0, 0.76653906543], [23.762711030000006, 0.0], [1.0, 0.76653906543], [0.0, 31.0]]],
                    [[[76, 82], [57, 1], [16, 19], [0, 19]], [[1.0, 1.0031225849], [57.0, 0.0], [1.0, 1.0031225849], [0.0, 56.822566712]], [[1.0, 1.0031225849], [0.0, 56.822566712], [1.0, 1.0031225849], [57.0, 0.0]]],
                    [[[96, 32], [15, 7], [16, 5], [14, 13]], [[1.0, 0.46875], [15.0, 0.0], [1.0, 0.46875], [0.0, 32.0]], [[1.0, 0.46875], [0.0, 32.0], [1.0, 0.46875], [15.0, 0.0]]],
                    [[[80, 24], [83, 2], [4, 15], [5, 3]], [[1.0, 40.0], [0.0, 2.0], [1.0, 40.0], [80.0, 0.0]], [[1.0, 40.0], [80.0, 0.0], [1.0, 40.0], [0.0, 2.0]]],
                    [[[98, 68], [59, 8], [15, 16], [5, 17]], [[1.0, 0.86764705882], [59.0, 0.0], [1.0, 0.86764705882], [0.0, 68.0]], [[1.0, 0.86764705882], [0.0, 68.0], [1.0, 0.86764705882], [59.0, 0.0]]],
                    [[[13, 91], [58, 6], [2, 13], [0, 8]], [[1.0, 0.01], [0.0, 6.0], [1.0, 0.01], [0.0600000000000005, 0.0]], [[1.0, 0.01], [0.060000000000002274, 0.0], [1.0, 0.01], [0.0, 6.0]]],
                    [[[3, 68], [98, 4], [7, 9], [5, 12]], [[1.0, 1.4411764706], [98.0, 0.0], [1.0, 1.4411764706], [0.0, 68.0]], [[1.0, 1.4411764706], [0.0, 68.0], [1.0, 1.4411764706], [98.0, 0.0]]],
                    [[[85, 14], [33, 22], [7, 14], [17, 15]], [[1.0, 0.01], [0.14000000000000057, 0.0], [1.0, 0.01], [0.0, 14.0]], [[1.0, 0.01], [0.0, 14.0], [1.0, 0.01], [0.14000000000000057, 0.0]]],
                    [[[4, 26], [48, 33], [18, 10], [2, 15]], [[1.0, 1.8461538462], [48.0, 0.0], [1.0, 1.8461538462], [0.0, 26.0]], [[1.0, 1.8461538462], [0.0, 26.0], [1.0, 1.8461538462], [48.0, 0.0]]],
                    [[[32, 69], [28, 94], [10, 1], [10, 14]], [[1.0, 0.01], [0.6899999999999977, 0.0], [1.0, 0.01], [0.0, 69.0]], [[1.0, 0.01], [0.0, 69.0], [1.0, 0.01], [0.6900000000000013, 0.0]]],
                    [[[71, 31], [3, 4], [9, 2], [0, 19]], [[1.0, 0.096774193548], [3.0, 0.0], [1.0, 0.096774193548], [0.0, 31.0]], [[1.0, 0.096774193548], [0.0, 31.0], [1.0, 0.096774193548], [3.0, 0.0]]]], dtype=tf.float32)
    t = tf.reshape(t, (t.shape[0], -1))
    output = custom_loss(t, t)
    print("output: ", output.numpy())
