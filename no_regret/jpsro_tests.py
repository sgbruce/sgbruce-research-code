import numpy as np
import tensorflow as tf
from jpsro import BidTradingEnv, solve_cce

def test_env():
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    env = BidTradingEnv(endowments, alphas)
    bids_to_test = [tf.convert_to_tensor([[[1, 1], [1, 1], [1, 2], [1, 1]], [[1, 1], [1, 1], [2, 1], [1, 1]]], dtype=tf.float32), \
                    tf.convert_to_tensor([[[1, 1], [2, 0], [1, 1], [0, 2]], [[1, 1], [0, 2], [1, 1], [2, 0]]], dtype=tf.float32), \
                    tf.convert_to_tensor([[[1, 1], [2, 2], [1, 1], [-2, -2]], [[1, 1], [-2, -2], [1, 1], [2, 2]]], dtype=tf.float32)]
    for bid in bids_to_test:
        _, rewards, _, info = env.step(bid)
        print("rewards: ", [reward.numpy() for reward in rewards])
        print("executed_bids: ", info["executed_bids"].numpy())
        print("allocations: ", [[alloc.numpy() for alloc in allocation] for allocation in info["allocations"]])
        print("valid_bids: ", info["valid_bids"])
        print("net_credit: ", info["net_credit"])
        print("\n\n\n")

def test_solver():
    L = tf.convert_to_tensor(np.array([1]))
    R = tf.convert_to_tensor(np.array([2]))
    meta_game_list = [[tuple([L, L]), [3, 3]], [tuple([L, R]), [1, 5]], [tuple([R, L]), [5, 6]], [tuple([R, R]), [7, 8]]]
    meta_game = [[joint, tf.convert_to_tensor(payoff)] for joint, payoff in meta_game_list]
    policy_sets = [[L, R], [L, R]]
    sigma = solve_cce(meta_game, 2, policy_sets)
    assert sigma[0][1] < 1e-4, "LL dominated, should have prob 0"
    assert sigma[1][1] < 1e-4, "LR dominated, should have prob 0"
    assert sigma[2][1] < 1e-4, "RL dominated, should have prob 0"
    assert sigma[3][1] > 1 - 1e-3, "RR dominates, should have prob 1"

    print("Dominant strategy example passed")

    meta_game_list = [[tuple([L, L]), [3, 4]], [tuple([L, R]), [1, 8]], [tuple([R, L]), [2, 6]], [tuple([R, R]), [7, 5]]]
    meta_game = [[joint, tf.convert_to_tensor(payoff)] for joint, payoff in meta_game_list]
    sigma = solve_cce(meta_game, 2, policy_sets)
    print([sigma[i][1] for i in range(len(sigma))])

if __name__ == "__main__":
    #test_env()
    test_solver()