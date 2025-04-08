from itertools import product

def get_num_strategies(num_goods):
    num_strategies = 0
    for buy_quantities in product(range(10), repeat=num_goods):
        for buy_price in product(range(10), repeat=num_goods):
            for sell_quantities in product(range(10), repeat=num_goods):
                for sell_price in product(range(10), repeat=num_goods):
                    num_strategies += 1
    return num_strategies

print(get_num_strategies(2))