# Implementation of CCE Estimation in a Bid-Based Trading Environment (TensorFlow)
import numpy as np
import tensorflow as tf
import cvxpy as cp
import itertools

# === Custom Trading Environment ===
class BidTradingEnv:
    def __init__(self, endowments, cobb_douglas_exponents):
        self.num_goods = 2
        self.endowments = endowments  # list of shape (num_players, num_goods)
        self.num_players = len(endowments)
        self.alphas = cobb_douglas_exponents  # list of shape (num_players, num_goods)
        self.reset()

    def reset(self):
        return np.zeros((len(self.endowments), self.num_goods))  # placeholder obs

    def step(self, bids: tf.Tensor):
        # bids = list of (buy_price, buy_qty, sell_price, sell_qty) * num_goods per agent
        executed_bids = self.match_trades(bids)
        # build allocations as purchase amount + endowment - sell amount
        allocations = []
        for i in range(self.num_players):
            allocation = []
            for j in range(self.num_goods):
                allocation.append(executed_bids[i, 1, j] + self.endowments[i, j] - executed_bids[i, 3, j])
            allocations.append(allocation)

        net_credit = []
        for i in range(self.num_players):
            net_credit.append(np.sum(executed_bids[i, 3] * executed_bids[i, 2] - executed_bids[i, 1] * executed_bids[i, 0]))

        rewards = self.compute_utilities(allocations, net_credit)

        valid_bids = []
        for i in range(self.num_players):
            bid = bids[i].numpy()
            if np.any(bid < 0) or np.any((self.endowments[i] - bid[3]) < 0):
                valid_bids.append(False)
            else:
                valid_bids.append(True)

        return None, rewards, True, {"executed_bids": executed_bids, "allocations": allocations, "valid_bids": valid_bids, "net_credit": net_credit}

    def match_trades(self, bids):
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
                if buy_quantity <= 0:
                    buy_index += 1
                    if buy_index < buy_ordering.shape[0]:
                        buyer_index = buy_ordering[buy_index]
                        buy_price = bids[buyer_index][0][good]
                        buy_quantity = bids[buyer_index][1][good]
                if sell_quantity <= 0:
                    sell_index += 1
                    if sell_index < sell_ordering.shape[0]:
                        seller_index = sell_ordering[sell_index]
                        sell_price = bids[seller_index][2][good]
                        sell_quantity = bids[seller_index][3][good]

                # Implement proportional rationing if needed
                # (This part can be expanded based on specific rules for rationing)
        return output_bids

    def compute_utilities(self, allocations, net_credit):
        utilities = []
        for i, alloc in enumerate(allocations):
            a1, a2 = self.alphas[i]
            utility = (alloc[0] ** a1) * (alloc[1] ** a2) + 0.5 * min(0, net_credit[i])
            utilities.append(utility)
        return utilities

    def evaluate_policy_tuple(self, policies):
        obs = self.reset()
        _, rewards, _, _ = self.step(tf.convert_to_tensor(policies, dtype=tf.float32))
        return rewards
    
    def create_basic_policy(self, player_idx):
        base_strat = [np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 2, np.random.uniform(0, 1, [2]), self.endowments[player_idx] / 2]
        return tf.convert_to_tensor(base_strat, dtype=tf.float32)
    
# === Policy Representation === maybe change this to a more concrete model
class NNPolicy():
    def __init__(self, obs_dim, act_dim):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(act_dim)
        ])
        self.trainable_variables = self.model.trainable_variables

    def __call__(self, obs):
        return self.model(obs)

def create_policy_model_nn(obs_dim, act_dim):
    return NNPolicy(obs_dim, act_dim)



# === Train Best Response ===
def train_best_response_nn(env: BidTradingEnv, policy_set_opponents, joint_distribution,player_idx, obs_dim, act_dim):
    br_policy = create_policy_model_nn(obs_dim, act_dim)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    for epoch in range(50):
        obs = env.reset()
        done = False
        while not done:
            with tf.GradientTape() as tape:
                logits = br_policy(tf.convert_to_tensor([obs[player_idx]], dtype=tf.float32))
                action = tf.squeeze(logits).numpy()
                actions = []
                for i, pi in enumerate(policy_set_opponents):
                    actions.append(tf.squeeze(pi(obs[i:i+1])).numpy() if i != player_idx else action)
                _, rewards, done, _ = env.step(actions)
                loss = -rewards[player_idx]  # maximize own utility
            grads = tape.gradient(loss, br_policy.trainable_variables)
            optimizer.apply_gradients(zip(grads, br_policy.trainable_variables))
    return br_policy

def train_best_response(env: BidTradingEnv, joint_distribution, player_idx, endowment, debug = False):
    def sample_joint_strategy(joint_distribution):
        keys = [joint_distribution[i][0] for i in range(len(joint_distribution))]
        probabilities = [joint_distribution[i][1] for i in range(len(joint_distribution))]
        selected_key = np.random.choice([i for i in range(len(joint_distribution))], p=probabilities)
        return keys[selected_key]
    
    # Step 1: Sample opponent bid profiles
    opponent_samples = []
    for _ in range(100):
        joint = sample_joint_strategy(joint_distribution)  # list of agent policies
        opp_bids = tf.convert_to_tensor([pi for i, pi in enumerate(joint) if i != player_idx], dtype=tf.float32)
        opponent_samples.append(opp_bids)
    
    if debug:
        print("opponent_samples generated: ", len(opponent_samples))
    
    # Step 2: Optimize agent's bid
    agent_bid = tf.Variable(tf.random.uniform([4, 2], maxval=[[1, 1], endowment, [1, 1], endowment]), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
    
    for epoch in range(200):  # optimization steps
        with tf.GradientTape() as tape:
            total = 0
            for opp_bids in opponent_samples:
                all_bids = tf.concat([opp_bids[:player_idx], tf.convert_to_tensor([agent_bid], dtype=tf.float32), opp_bids[player_idx:]], axis=0)
                _, utils, _, _ = env.step(all_bids)
                total += utils[player_idx]
            loss = -total / len(opponent_samples)  # maximize expected utility
        grads = tape.gradient(loss, [agent_bid])
        optimizer.apply_gradients(zip(grads, [agent_bid]))
        if debug and epoch % 10 == 0:
            print("epoch: ", epoch, "loss: ", loss)
            print("agent_bid: ", agent_bid.numpy())
    
    return agent_bid.numpy()

# === Meta-Game Construction ===
def build_meta_game(env: BidTradingEnv, policy_sets):
    joint_policies = list(itertools.product(*policy_sets))
    meta_game = []
    for i, joint in enumerate(joint_policies):
        payoff = env.evaluate_policy_tuple(joint)
        meta_game.append([joint, payoff])
    return meta_game

# === CCE Solver ===
def solve_cce(meta_game, num_players, policy_sets):
    joint_strats = [meta_game[i][0] for i in range(len(meta_game))]
    joint_payoffs = np.array([meta_game[i][1] for i in range(len(meta_game))])
    num_strats = len(joint_strats)
    
    sigma = cp.Variable(num_strats)
    objective = cp.Maximize(-0.5 * cp.quad_form(sigma, np.eye(num_strats)))
    constraints = [cp.sum(sigma) == 1, sigma >= 0]
    for p in range(num_players):
        for alt_pi in policy_sets[p]:
            constraint_sum = 0
            for i, pi in enumerate(joint_strats): # i is the index of the joint strategy, pi is the joint strategy
                pi_alt = list(pi)
                original_payoff = joint_payoffs[i][p] # get the payoff for the current strategy for player p
                if alt_pi.numpy().all() == pi[p].numpy().all(): # if the alternate strategy is the same as the current strategy, skip
                    alt_payoff = original_payoff
                else:
                    pi_alt[p] = alt_pi # update the current strategy for player p
                    if tuple(pi_alt) not in joint_strats: # if the alternate strategy is not in the meta game, skip
                        continue
                    alt_payoff = joint_payoffs[joint_strats.index(tuple(pi_alt))][p] # get the payoff for the alternate strategy for player p
                constraint_sum += sigma[i] * (alt_payoff - original_payoff) # add the constraint for the current strategy
            constraints.append(constraint_sum <= 1e-4) # add the constraint to the list of constraints
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return [(joint_strats[i], sigma.value[i]) for i in range(len(joint_strats))]


def run_training(debug = False):
    # === Main Loop ===
    num_players = 2
    obs_dim, goods, act_per_good = 2, 2, 4
    act_dim = goods * act_per_good
    endowments = np.array([[4, 4], [4, 4]])
    alphas = np.array([[0.75, 0.25], [0.25, 0.75]])
    # optimal outcome is p=[1,1], x=[[6,2],[2,6]]
    env = BidTradingEnv(endowments, alphas)
    policy_sets = [[env.create_basic_policy(p)] for p in range(num_players)]
    meta_game = build_meta_game(env, policy_sets)
    sigma = solve_cce(meta_game, num_players, policy_sets)

    if debug:
        print("initial Configuration:")
        print("policy_sets: ", [[pi.numpy() for pi in policy_sets[p]] for p in range(num_players)])
        print("meta_game: ", [[i, payoff] for i, [joint, payoff] in enumerate(meta_game)])
        print("sigma: ", [[i, sigma[i][1]] for i in range(len(sigma))])

    for epoch in range(10):
        for p in range(num_players):
            new_pi = train_best_response(env, sigma, p, endowments[p], debug=debug)
            if debug:
                print("new policy for player ", p, ": ", new_pi)
            policy_sets[p].append(new_pi)
        meta_game = build_meta_game(env, policy_sets)
        sigma = solve_cce(meta_game, num_players, policy_sets)

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
if __name__ == "__main__":
    run_training(debug = True)
    # test_env()