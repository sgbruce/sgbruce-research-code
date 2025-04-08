import json
from dubey_regret import DubeyGame, DubeyPlayer, Strategy
import numpy as np
from gekko import GEKKO

def get_competitive_solution(endowments, preferences, debug=False):
    m = GEKKO(remote=False)

    num_players = len(endowments)
    num_goods = len(endowments[0])

    # Create parameters for endowments and preferences
    e = [[m.Param(endowments[i][j]) for j in range(num_goods)] for i in range(num_players)]
    p = [[m.Param(preferences[i][j]) for j in range(num_goods)] for i in range(num_players)]

    # Create variables for each player's allocation
    x = [[m.Var(value=0.5, lb=0) for _ in range(num_goods)] for _ in range(num_players)]

    # Create variables for prices
    prices = [m.Var(value=1, lb=0.01) for _ in range(num_goods)]  # Set lb>0 to avoid trivial zero prices
    m.Equation(prices[0] == 1) # remove degree of freedom

    # Budget constraints: each player's spending ≤ initial endowment value
    for i in range(num_players):
        m.Equation(sum(x[i][j] * prices[j] for j in range(num_goods)) 
                   <= sum(e[i][j] * prices[j] for j in range(num_goods)))

    # Market clearing condition: total allocation must equal total endowment
    for j in range(num_goods):
        m.Equation(sum(x[i][j] for i in range(num_players)) == sum(e[i][j] for i in range(num_players)))

    # Objective function: Maximize total weighted utility
    utility = sum(sum(x[i][j] * p[i][j] for j in range(num_goods)) for i in range(num_players))
    m.Obj(-utility)  # Minimize the negative to maximize

    # Solve the model
    m.solve(disp=debug)

    return [[xi.value for xi in x_row] for x_row in x], [p.value for p in prices]

def get_optimal_bids(endowments, x, prices, debug=False):
    s = [Strategy(prices, np.maximum(0, x[i] - endowments[i]), prices, np.maximum(0, endowments[i] - x[i])) for i in range(len(x))]
    if debug:
        print(s)
    return s

def get_training_data(game, debug=False):
        # generate random endowments and preferences
        endowments = np.random.randint(0, 100, size=(len(game.players), game.num_goods))
        preferences = np.random.randint(0, 20, size=(len(game.players), game.num_goods))
        context = np.concatenate([endowments, preferences], axis=0)
        x, prices = get_competitive_solution(endowments, preferences)
        x = np.moveaxis(np.array(x), 2, 0)[0]
        prices = np.moveaxis(np.array(prices), 1, 0)[0]
        if debug:
            print(x)
            print(prices)
        optimal_strategies = get_optimal_bids(endowments, x, prices, debug)
        # use these random endowments to find optimal player strategy
        #optimal_strategies = np.array([self.get_optimal_strategy(context[i], i) for i in range(len(context))])
        return context, optimal_strategies

def gen_training_data():
    game = DubeyGame(num_goods=2)
    game.add_player(DubeyPlayer(0, [1, 1]))
    game.add_player(DubeyPlayer(0, [1, 1]))
    iter = 0
    max_iter = 100000
    with open('training_data_2_2.json', 'w') as f:
        f.write('[\n')
    while True:
        try:
            context, optimal_strategies = get_training_data(game)
            datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
            # print(datapoint)
            with open('training_data_2_2.json', 'a') as f:
                json.dump(datapoint, f)
                if iter < max_iter:
                    f.write(',\n')
        except Exception as e:
            print(e)
        iter += 1
        if iter % 500 == 0:
            print(iter)
        if iter >= max_iter:
            break
    with open('training_data_2_2.json', 'a') as f:
        f.write(']')
    
    # game.add_player(DubeyPlayer(0, [1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_3_2.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break
    
    # game.add_player(DubeyPlayer(0, [1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_4_2.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break

    # game = DubeyGame(num_goods=3)
    # game.add_player(DubeyPlayer(0, [1, 1, 1]))
    # game.add_player(DubeyPlayer(0, [1, 1, 1]))
    # game.add_player(DubeyPlayer(0, [1, 1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_3_3.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break

    # game.add_player(DubeyPlayer(0, [1, 1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_4_3.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break
    
    # game.add_player(DubeyPlayer(0, [1, 1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_5_3.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break
    
    # game = DubeyGame(num_goods=4)
    # game.add_player(DubeyPlayer(0, [1, 1, 1, 1]))
    # game.add_player(DubeyPlayer(0, [1, 1, 1, 1]))
    # game.add_player(DubeyPlayer(0, [1, 1, 1, 1]))
    # game.add_player(DubeyPlayer(0, [1, 1, 1, 1]))
    # iter = 0
    # while True:
    #     try:
    #         context, optimal_strategies = get_training_data(game)
    #         datapoint = [context.tolist(), [[list(s.buy_price), list(s.buy_quantity), list(s.sell_price), list(s.sell_quantity)] for s in optimal_strategies]]
    #         # print(datapoint)
    #         with open('training_data_4_4.json', 'a') as f:
    #             json.dump(datapoint, f)
    #             f.write(',\n')
    #     except Exception as e:
    #         print(e)
    #     iter += 1
    #     if iter % 500 == 0:
    #         print(iter)
    #     if iter > 10000:
    #         break

if __name__ == "__main__":
    gen_training_data()