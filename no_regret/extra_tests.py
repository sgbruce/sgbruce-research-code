from dubey_regret import DubeyGame, DubeyPlayer, Strategy
from oracle import NNOracle
import numpy as np
import tensorflow as tf

def get_strategy_vector(context, strategy1, strategy2):
    flat_strategy1 = strategy1.as_flat_array()
    flat_strategy2 = strategy2.as_flat_array()
    vec = tf.reshape(np.concatenate([context, flat_strategy1, flat_strategy2], axis=0), (1, -1))
    return tf.cast(vec, tf.float32)

def loss_outcome(loss_fn, y_vector_1, y_vector_2, label_1, label_2, expected_loss, debug = False):
    mean_loss = loss_fn(y_vector_1, y_vector_2).numpy()
    if debug:
        print(f'mean_loss of ({label_1}, {label_2}): {mean_loss}')
    assert mean_loss == expected_loss, f'Loss for ({label_1}, {label_2}) should be {expected_loss}, but got {mean_loss}'

def test_loss_fn(oracle, debug = False):
    # get the loss function
    oracle_loss = oracle.get_custom_loss(debug = debug)
    # get the context
    endowments = np.array([player.goods for player in game.players])
    utilities = np.array([[2, 0], [0, 2]])
    context = np.reshape(np.concatenate([endowments, utilities]), -1)

    # get the zero strategy
    zero_strategy = Strategy(np.zeros(game.num_goods), np.zeros(game.num_goods), np.zeros(game.num_goods), np.zeros(game.num_goods))
    zero_transfer = get_strategy_vector(context, zero_strategy, zero_strategy)

    # get the strategy where one good is transferred from player 0 to player 1
    strat_0 = Strategy(np.array([1, 0]), np.array([1, 0]), np.array([0, 0]), np.array([0, 0]))
    strat_1 = Strategy(np.array([0, 0]), np.array([0, 0]), np.array([1, 0]), np.array([1, 0]))
    one_good_transfer = get_strategy_vector(context, strat_0, strat_1)

    strat_0 = Strategy(np.array([1, 0]), np.array([1, 0]), np.array([0, 1]), np.array([0, 1]))
    strat_1 = Strategy(np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]))
    two_good_transfer = get_strategy_vector(context, strat_0, strat_1)

    strat_0 = Strategy(np.array([1, 0]), np.array([2, 0]), np.array([0, 1]), np.array([0, 2]))
    strat_1 = Strategy(np.array([0, 1]), np.array([0, 1]), np.array([1, 0]), np.array([1, 0]))
    two_good_transfer_unbalanced = get_strategy_vector(context, strat_0, strat_1)

    strat_0 = Strategy(np.array([5, 0]), np.array([5, 0]), np.array([0, 5]), np.array([0, 5]))
    strat_1 = Strategy(np.array([0, 5]), np.array([0, 5]), np.array([5, 0]), np.array([5, 0]))
    two_good_transfer_extra = get_strategy_vector(context, strat_0, strat_1)

    # get the loss for the zero strategy with itself
    loss_outcome(oracle_loss, zero_transfer, zero_transfer, "zero strategy", "zero strategy", 0, debug = debug)
    # get the loss for the one good transfer
    loss_outcome(oracle_loss, one_good_transfer, one_good_transfer, "one good transfer", "one good transfer", 0, debug = debug)
    loss_outcome(oracle_loss, one_good_transfer, zero_transfer, "one good transfer", "zero strategy", utilities[0][0], debug = debug)
    # get the strategy where two good is transferred from player 0 to player 1
    loss_outcome(oracle_loss, two_good_transfer, two_good_transfer, "two good transfer", "two good transfer", 0, debug = debug)
    loss_outcome(oracle_loss, two_good_transfer, zero_transfer, "two good transfer", "zero strategy", utilities[0][0] + utilities[1][1], debug = debug)
    loss_outcome(oracle_loss, two_good_transfer, one_good_transfer, "two good transfer", "one good transfer", utilities[1][1], debug = debug)
    # get the loss for the two good transfer unbalanced and extra
    loss_outcome(oracle_loss, two_good_transfer, two_good_transfer_unbalanced, "two good transfer unbalanced", "two good transfer", 0, debug = debug)
    loss_outcome(oracle_loss, two_good_transfer_extra, zero_transfer, "two good transfer extra", "zero strategy", 5 * (utilities[0][0] + utilities[1][1]), debug = debug)
    loss_outcome(oracle_loss, two_good_transfer_extra, two_good_transfer, "two good transfer extra", "two good transfer", 4 * (utilities[0][0] + utilities[1][1]), debug = debug)
    loss_outcome(oracle_loss, two_good_transfer_extra, two_good_transfer_unbalanced, "two good transfer extra", "two good transfer unbalanced", 4 * (utilities[0][0] + utilities[1][1]), debug = debug)


def predict_test(oracle):
    nn_oracle.train_nn(reduced = True)
    utilities = np.array([[1, 0], [0, 1]])
    context, bids = oracle.predict_optimal_strategies(utilities)
    strategies = [Strategy(bid[0], bid[1], bid[2], bid[3]) for bid in bids]
    print("context: ", context)
    for index, strategy in enumerate(strategies):
        print("strategy: ", strategy)
        print("utility: ", game.get_utility(index, strategy))


if __name__ == "__main__":
    game = DubeyGame(2)
    game.add_player(DubeyPlayer(1, [0, 10]))
    game.add_player(DubeyPlayer(1, [10, 0]))
    nn_oracle = NNOracle(game)
    
    # predict_test(nn_oracle, utilities)
    test_loss_fn(nn_oracle, debug = True)