'''
This file contains the oracle for the continuous Dubey game.
The file contains the regression oracle as well as the sampling oracle.

The simplified regression oracle is a full information, determinstic oracle, 
meaning that instead of an online algorithm, it is a batch algorithm that 
receives all the limit orders at once and then computes the true outcome. 

Likewise, it can also efficiently compute an optimal strategy for a player given 
the strategies of the other players and the player's preferences.

The sampling oracle is an online algorithm that uses sampling to return a 
random strategy 
'''

import numpy as np
import json
from dubey_regret import DubeyGame, DubeyPlayer, Tensor_Strategy, Strategy
import tensorflow as tf
from tensorflow import keras
from gekko import GEKKO
from datetime import datetime

BETA = 10
OVERSOLD_PENALTY = 10000
EXCESS_BID_PENALTY = 0.1
MSE_SCALE = 1

class SimpleOracle:
    def __init__(self, game: DubeyGame): # BUILD A NN THAT, GIVEN ENDOWMENTS AND UTILTIES, FINDS OPTIMAL BIDS
        # CAN I MAKE INPUTS THAT HAVE A KNOWN MAX TOTAL WELFARE? THEN RANDOMIZE THE STARTING ALLOCATIONS, 
        # AND SEE IF THE NN CAN FIND THE TRADES TO INDUCED THAT MAX WELFARE?
        self.game = game

    def get_optimal_strategy(self, player_index, other_strategies):
        pass

    def get_true_outcome(self, strategies):
        pass

class NNOracle:
    def __init__(self, game: DubeyGame):
        self.game = game
        self.num_inputs = 2 * self.game.num_goods * len(self.game.players)
        self.num_outputs = 4 * self.game.num_goods * len(self.game.players)

        # Initialize the model
        normalizer = keras.layers.Normalization(axis=-1)
        input_layer = keras.layers.Input(shape=(self.num_inputs,))
        x = normalizer(input_layer)
        x = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(512, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(1024, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(2048, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(1024, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(512, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        x = keras.layers.Dense(256, activation=keras.layers.LeakyReLU(alpha=0.1))(x)
        output_layer = keras.layers.Dense(self.num_outputs, activation=keras.layers.ReLU())(x)
        combined_output = keras.layers.Concatenate()([input_layer, output_layer])
        self.nn = keras.Model(inputs=input_layer, outputs=combined_output)
    
    def get_training_data(self):
        print("reading training data")
        training_data_file = f"training_data_{len(self.game.players)}_{self.game.num_goods}_cobb.json"
        with open(training_data_file, 'r') as f:
            training_data = json.load(f)
        print("training data loaded")
        training_data_x = []
        training_data_y = []
        for entry in training_data:
            x, y = entry
            training_data_x.append(x)
            training_data_y.append(y)
        training_data_x = np.array(training_data_x).reshape(-1, self.num_inputs)
        training_data_y = np.array(np.concatenate(training_data_y)).reshape(-1, self.num_outputs)
        training_data_y = np.concatenate([training_data_x, training_data_y], axis=1)
        return training_data_x, training_data_y
    
    def get_custom_loss(self, debug = False):
        num_players = len(self.game.players)
        num_goods = self.game.num_goods
        def custom_loss(y_true, y_pred):
            def per_sample_loss(true, pred):
                # Use TensorFlow operations instead of NumPy
                true_context = tf.reshape(true[:self.num_inputs], (2, num_players, num_goods))
                pred_context = tf.reshape(pred[:self.num_inputs], (2, num_players, num_goods))
                
                endowments = true_context[0]
                preferences = true_context[1]
                
                true_bids = tf.reshape(true[self.num_inputs:], (num_players, 4, num_goods))
                pred_bids = tf.reshape(pred[self.num_inputs:], (num_players, 4, num_goods))

                # Create a mask where the sale quantity is greater than the endowment
                sale_quantity = tf.slice(pred_bids, [0, 3, 0], [-1, 1, -1])  # Sale quantity for each player and good
                endowments_reshaped = tf.reshape(endowments, (num_players, 1, num_goods))
                penalty = tf.nn.relu(sale_quantity - endowments_reshaped)
                total_oversold_amount = tf.reduce_sum(penalty)

                
                # Compute trade amounts using TensorFlow operations
                true_trade_amounts = self.game.compute_trade_amounts(true_bids)
                pred_trade_amounts = self.game.compute_trade_amounts(pred_bids)
                # if debug:
                #     print("true_trade_amounts: ", true_trade_amounts)
                #     print("pred_trade_amounts: ", pred_trade_amounts)

                def compute_utility(trade_amounts):
                    bought_prices = trade_amounts[:, 0]
                    bought_quantities = trade_amounts[:, 1]
                    sold_prices = trade_amounts[:, 2]
                    sold_quantities = trade_amounts[:, 3]
                    endowed_cash = tf.constant([player.money for player in self.game.players], dtype=tf.float32)
                    player_cash = tf.reduce_sum(sold_prices * sold_quantities - bought_prices * bought_quantities, axis=1) + endowed_cash
                    cash_penalty = tf.nn.softplus(-player_cash)
                    player_utilities = tf.reduce_sum(preferences * bought_quantities, axis=1)
                    utility = tf.reduce_sum(player_utilities - BETA * cash_penalty)
                    # if debug:
                    #     print("\n bought_quantities: ", bought_quantities)
                    #     print("player_utilities: ", player_utilities)
                    #     print("cash_penalty: ", cash_penalty)
                    #     print("utility: ", utility)
                    return utility
                
                def compute_excess_bid(bids, trade_amounts):
                    excess = bids - trade_amounts
                    # check that the excess is non-negative
                    # assert tf.reduce_sum(tf.where(excess > 0, tf.zeros_like(excess), excess)) == 0
                    # # check that the excess is zero for the prices
                    # assert tf.reduce_sum(excess[:,0,:]) == 0
                    # assert tf.reduce_sum(excess[:,2,:]) == 0
                    return tf.reduce_sum(excess)
                
                true_utility = compute_utility(true_trade_amounts)
                pred_utility = compute_utility(pred_trade_amounts)

                excess_bid = compute_excess_bid(pred_bids, pred_trade_amounts)

                mse_fallback = tf.reduce_mean(tf.square(true - pred))
                
                return true_utility - pred_utility + OVERSOLD_PENALTY * total_oversold_amount + MSE_SCALE * mse_fallback #+ EXCESS_BID_PENALTY * excess_bid

            # Use tf.map_fn with TensorFlow operations
            per_sample_losses = tf.map_fn(lambda x: per_sample_loss(x[0], x[1]), (y_true, y_pred), fn_output_signature=tf.float32)
            mean_loss = tf.reduce_mean(per_sample_losses)
            return mean_loss
        return custom_loss
    
    def get_custom_loss_vectorized(self, debug = False):
        num_players = len(self.game.players)
        num_goods = self.game.num_goods
        def custom_loss_vectorized(y_true, y_pred):
            # Split context and bids
            true_context = tf.reshape(y_true[:, :self.num_inputs], (-1, 2, num_players, num_goods))
            pred_context = tf.reshape(y_pred[:, :self.num_inputs], (-1, 2, num_players, num_goods))

            endowments = true_context[:, 0]
            preferences = true_context[:, 1]

            true_bids = tf.reshape(y_true[:, self.num_inputs:], (-1, num_players, 4, num_goods))
            pred_bids = tf.reshape(y_pred[:, self.num_inputs:], (-1, num_players, 4, num_goods))

            # OVERSOLD PENALTY
            sale_quantity = pred_bids[:, :, 3]  # shape (batch, num_players, num_goods)
            oversold_amount = tf.nn.relu(sale_quantity - endowments)
            total_oversold = tf.reduce_sum(oversold_amount, axis=[1, 2])  # shape (batch,)

            # TRADE AMOUNTS (batch compute)
            true_trade_amounts = self.game.compute_trade_amounts_vectorized(true_bids)  # should be batched
            pred_trade_amounts = self.game.compute_trade_amounts_vectorized(pred_bids)  # should be batched

            def compute_utility(trade_amounts):
                bought_prices = trade_amounts[:, :, 0]
                bought_quantities = trade_amounts[:, :, 1]
                sold_prices = trade_amounts[:, :, 2]
                sold_quantities = trade_amounts[:, :, 3]
                endowed_cash = tf.constant([player.money for player in self.game.players], dtype=tf.float32)
                endowed_cash = tf.reshape(endowed_cash, (1, -1))  # shape (1, num_players)
                cash = tf.reduce_sum(sold_prices * sold_quantities - bought_prices * bought_quantities, axis=2) + endowed_cash
                cash_penalty = -tf.nn.softplus(-cash)
                player_utils = tf.reduce_sum(preferences * bought_quantities, axis=2)
                return tf.reduce_sum(player_utils - BETA * cash_penalty, axis=1)  # shape (batch,)

            assert_not_nan = tf.debugging.assert_all_finite(pred_trade_amounts, "pred_trade_amounts contains NaNs or Infs")
            with tf.control_dependencies([assert_not_nan]):
                true_util = compute_utility(true_trade_amounts)
                pred_util = compute_utility(pred_trade_amounts)

            # EXCESS BID (if needed)
            # excess_bid = tf.reduce_sum(pred_bids - pred_trade_amounts, axis=[1,2,3])

            # fallback mse
            mse_loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

            total_loss = (true_util - pred_util) + OVERSOLD_PENALTY * total_oversold + MSE_SCALE * mse_loss
            return tf.reduce_mean(total_loss)
        return custom_loss_vectorized

            
    
    # Convert to Strategy objects if needed, or use TensorFlow operations
                # true_bids = [Tensor_Strategy(true_bids[i][0], true_bids[i][1], true_bids[i][2], true_bids[i][3]) for i in range(num_players)]
                # pred_bids = [Tensor_Strategy(pred_bids[i][0], pred_bids[i][1], pred_bids[i][2], pred_bids[i][3]) for i in range(num_players)]

    def train_nn(self, reduced = False):
        # get the training context and optimal market value
        training_data_x, training_data_y = self.get_training_data()
        if reduced:
            training_data_x = training_data_x[0:len(training_data_x) // 5]
            training_data_y = training_data_y[0:len(training_data_y) // 5]
        # Split the training data into train and validation splits 80/20
        split_index = int(0.8 * len(training_data_x))
        train_x, val_x = training_data_x[:split_index], training_data_x[split_index:]
        train_y, val_y = training_data_y[:split_index], training_data_y[split_index:]
        # train the nn, need to fix loss function
        print(train_x.shape)
        print(train_y.shape)

        self.nn.compile(loss=self.get_custom_loss())
        epochs = 5 if reduced else 20
        self.nn.fit(train_x, train_y, epochs=epochs, batch_size=16, validation_data=(val_x, val_y))

    def train_nn_no_custom_loss(self, reduced = False):
        # get the training context and optimal market value
        training_data_x, training_data_y = self.get_training_data()
        if reduced:
            training_data_x = training_data_x[0:len(training_data_x) // 1]
            training_data_y = training_data_y[0:len(training_data_y) // 1]
        # Split the training data into train and validation splits 80/20
        # split_index = int(0.8 * len(training_data_x))
        # train_x, val_x = training_data_x[:split_index], training_data_x[split_index:]
        # train_y, val_y = training_data_y[:split_index], training_data_y[split_index:]
        # train the nn, need to fix loss function
        print(training_data_x.shape)
        print(training_data_y.shape)

        self.nn.compile(optimizer='adam', loss='mse')
        epochs = 2 if reduced else 50
        self.nn.fit(training_data_x, training_data_y, epochs=epochs, batch_size=1, validation_split=0.2)

    '''
    context is a list of the players' strategies and endowments
    Use NN to predict the optimal strategy for the player
    '''
    def predict_optimal_strategies(self, utilities):
        context_unflattened = np.array([[player.goods for player in self.game.players], utilities])
        context = np.reshape(context_unflattened, (1, -1))
        output = self.nn.predict(context)
        context_output = output[0][:self.num_inputs]
        strategy_output = output[0][self.num_inputs:]
        bids_split = np.reshape(strategy_output, (len(self.game.players), 4, self.game.num_goods))
        return context_output, bids_split

    def get_true_outcome(self, strategies):
        pass

    def export_nn(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.nn.save(f'nn_{timestamp}.h5')

if __name__ == "__main__":
    # nn_oracle = NNOracle(game)
    # nn_oracle.train_nn()
    print("starting")
    np.random.seed(42)
    game = DubeyGame(num_goods=2)
    game.add_player(DubeyPlayer(0, [10, 10]))
    game.add_player(DubeyPlayer(0, [10, 10]))
    nn_oracle = NNOracle(game)
    print("game created")
    nn_oracle.train_nn_no_custom_loss(reduced = True)
    print("nn trained")
    # nn_oracle.export_nn()
    utilities = np.array([[0.3, 0.7], [0.7, 0.3]])
    context, bids = nn_oracle.predict_optimal_strategies(utilities)
    strategies = [Strategy(bid[0], bid[1], bid[2], bid[3]) for bid in bids]
    print("context: ", context)
    for index, strategy in enumerate(strategies):
        print("strategy: ", strategy)
        print("utility: ", game.get_utility(index, strategy))


    # x, prices = get_competitive_solution()
    # print(np.moveaxis(np.array(x), 2, 0)[0])
    # print(prices)