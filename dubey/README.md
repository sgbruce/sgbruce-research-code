# Dubey Directory

This directory contains the implementation of a market game based on the Dubey model. The Dubey model is a theoretical framework for analyzing trading strategies and market dynamics among traders with different endowments and utility functions. More concrete definitions can be found in the no_regret directory, however here is the initial implementations of the Dubey model, a strategy, and a player.

## Files

### `game.py`
This file contains the core implementation of the Dubey market game. It defines the `strategy` class, which represents the bids to buy and sell goods, and the `DubeyPlayer` class, which represents a player in the market with a certain amount of money and goods, as well as the `DubeyGame` class, which represents the market game.

### `price_given.py`
This file provides an implementation of the Dubey market game where prices are assumed to be given. This simplification allows for a more straightforward implementation of the game mechanics. It 
otherwise is similar to the `game.py` file.

