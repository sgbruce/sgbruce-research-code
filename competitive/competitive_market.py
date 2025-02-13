'''
This file provides a class to represent a competitive market. 
Each market should have a set of agents, each with a Cobb-Douglas utility function.
Each agent also has a set of endowments.
'''
import numpy as np

def print_array(array):
    print("".join([str(array[i].round(2)) + "\n" for i in range(len(array))]))

class CompetitiveMarket:
    def __init__(self, endowments, alphas):
        assert len(endowments) == len(alphas)
        self.endowments = np.array(endowments)
        self.alphas = np.array(alphas)
        self.outcomes = np.array(endowments)
        self.prices = None

    def update_market(self, outcomes, prices):
        self.outcomes = np.array(outcomes)
        self.prices = np.array(prices)

    def print_market(self):
        print("Market Results:")
        print("There are ", len(self.endowments), " agents in the market.")
        print("The initial endowments are:")
        print_array(self.endowments)
        print("The alphas are: ")
        print_array(self.alphas)
        print("The outcomes are: ")
        print_array(self.outcomes)
        print("The prices are: ")
        print_array(self.prices)
