import numpy as np
from learner import NoRegretLearner, DubeyLearner
from dubey_regret import DubeyGame, get_all_strategies
import time
import matplotlib.pyplot as plt

def test_no_regret_learner():
    strategies = ['A', 'B', 'C']
    learner = NoRegretLearner(strategies)
    
    # Test initial probabilities
    assert np.allclose(learner.probabilities, np.ones(len(strategies)) / len(strategies)), "Initial probabilities are incorrect"
    
    # Test strategy sampling
    sampled_strategy = learner.sample_strategy()
    assert sampled_strategy in strategies, "Sampled strategy is not in the list of strategies"
    
    # Test weight update
    cost_vector = np.array([1, 0, 0.5])
    learner.update(cost_vector, eta=0.1)
    print(learner.weights)
    expected_weights = (0.9) ** cost_vector
    expected_probabilities = expected_weights / expected_weights.sum()
    assert np.allclose(learner.weights, expected_weights), "Weights after update are incorrect"
    assert np.allclose(learner.probabilities, expected_probabilities), "Probabilities after update are incorrect"

    print("All tests passed!")


if __name__ == "__main__":
    # test_no_regret_learner()
    #print(list(range(1,2)))
    strategies = get_all_strategies(2, restricted=False)

    game = DubeyGame(2)
    learner_0 = DubeyLearner(strategies, 0, [5, 5])
    learner_1 = DubeyLearner(strategies, 0, [5, 5])
    game.add_player(learner_0)
    game.add_player(learner_1)

    start_time = time.time()
    for i in range(2000):
        iter_time = time.time()
        game.run_mechanism(debug=False)
        if i%100==0:
            print(f"Iteration {i} took {time.time() - iter_time} seconds")
    print(f"Total time: {time.time() - start_time} seconds")

    title = "Dubey Game"

    print("Final probabilities:")
    for i, s in enumerate(strategies):
        if learner_0.probabilities[i] > 0.00001:
            print(s, learner_0.probabilities[i].round(3), "Valid: ", learner_0.is_valid_strategy(s))
    print("-----")
    for i, s in enumerate(strategies):
        if learner_1.probabilities[i] > 0.00001:
            print(s, learner_1.probabilities[i].round(3), "Valid: ", learner_1.is_valid_strategy(s))

    learner_0_avg_prob = np.mean(learner_0.all_probabilities, axis=0)# np.zeros((len(learner_0.all_probabilities), len(learner_0.strategies)))
    learner_1_avg_prob = np.mean(learner_1.all_probabilities, axis=0)# np.zeros((len(learner_1.all_probabilities), len(learner_1.strategies)))
    # for i in range(1, len(learner_0.all_probabilities)):
    #     learner_0_avg_prob[i, :] = np.mean(learner_0.all_probabilities[:i, :], axis=0)
    #     learner_1_avg_prob[i, :] = np.mean(learner_1.all_probabilities[:i, :], axis=0)
    
    print("Final average probabilities:")
    for i, s in enumerate(strategies):
        if learner_0_avg_prob[i] > 0.00001:
            print(s, learner_0_avg_prob[i].round(3), "Valid: ", learner_0.is_valid_strategy(s))
    print("-----")
    for i, s in enumerate(strategies):
        if learner_1_avg_prob[i] > 0.00001:
            print(s, learner_1_avg_prob[i].round(3), "Valid: ", learner_1.is_valid_strategy(s))

    # iterations = np.arange(len(learner_0.all_probabilities))
    # plt.figure(figsize=(12, 6))
    # plt.suptitle(title, fontsize=16)
    # plt.subplots_adjust(top=0.88)

    # plt.subplot(2, 2, 1)
    # for i in range(len(learner_0.strategies)):
    #     plt.plot(iterations, learner_0.all_probabilities[:, i], label=str(learner_0.strategies[i]))
    # plt.xlabel('Iterations')
    # plt.ylabel('Probability')
    # plt.title('Learner\'s Strategy Probabilities')
    # plt.legend()

    # plt.subplot(2, 2, 2)
    # for i in range(len(learner_1.strategies)):
    #     plt.plot(iterations, learner_1.all_probabilities[:, i], label=str(learner_1.strategies[i]))
    # plt.xlabel('Iterations')
    # plt.ylabel('Probability')
    # plt.title('Learner\'s Strategy Probabilities')
    # plt.legend()

    # plt.subplot(2, 2, 3)
    # for i in range(len(learner_0.strategies)):
    #     plt.plot(iterations, learner_0_avg_prob[:, i], label=str(learner_0.strategies[i]))
    # plt.xlabel('Iterations')
    # plt.ylabel('Probability')
    # plt.title('Learner\'s Time Average Strategy Probabilities')
    # plt.legend()

    # plt.subplot(2, 2, 4)
    # for i in range(len(learner_1.strategies)):
    #     plt.plot(iterations, learner_1_avg_prob[:, i], label=str(learner_1.strategies[i]))
    # plt.xlabel('Iterations')
    # plt.ylabel('Probability')
    # plt.title('Learner\'s Time Average Strategy Probabilities')
    # plt.legend()

    # plt.tight_layout()
    # plt.show()
    