# Computing Market Equilibria Code Repository 
By Sam Bruce s20sbruce@gmail.com

This repository contains the research code generated during my work with professor Townsend and is organized into several directories, each corresponding to specific equilibrium computation tools and strategies. The vast majority of these files are written in python, except for the smart contract helper code (typescript) and the contract code itself (solidity). 

To run the code, you will need a working python installation (I used python 3.8 for most of it, but any later version should work). The necessary python packages are listed in `requirements.txt` in the base directory. The solidity contracts can be imported into the remix IDE on a web browserand run there successfully using the code in the `contracts` and `scripts` directories.

The notes that accompany this repository can be found at this link: https://succinct-harp-6c3.notion.site/sgbruce-Research-Notes-df6c371c11154fdbb2cdb8eca414ad63. These notes are comprehensive of the topics covered in this repository, as well as much more additional notes on the topics of correlated equilibria, coarse correlated equilibria, and no-regret learning. Additionally, the completed thesis document summarizing this research can be found at the same link.

Finally, each directory has it's own readme file to understand each file in the directory, please refer to those files for a more detailed explanation of the code layout.

## Directory Overview

### `competitive/`
This directory is mostly an artifact of the work towards a correlated equilibrium of the market game. Computing the competitive equilibrium directly is proven to be PPAD-complete, so the efforts in this directory are either infeasible or have large assumptions. There are some basic components of market representation that are reused in other directories, however the contents here are largeley unfinished.

### `contracts/`
This directory contains Solidity smart contracts intended for blockchain deployment, primarily focusing on addressing the multi-principal agent problem. It features implementations for both centralized tracking of commitments and decentralized systems utilizing colored coins. The contracts aim to apply economic theories, such as those from Bernheim and Winston's "Common Agency," to effectively manage agent interactions and determine payouts. Lastly, an on-chain correlated equilibrium solver is included, though unfinished.

### `scripts/`
This directory contains helper code for running and testing the solidity smart contracts in the contracts directory. The files in this directory are in typescript, and handle connectionto blockchain environments as well as deploying contracts to the blockchain, and finally testing said deployed contracts.

### `correlated/`
The `correlated` directory includes explicit linear programming code for computing correlated equilibria. It includes mutliple ways of computing the contraint matrices, as well as tests proving the correctness of the equilibria and equilibrium selection.

### `dubey/`
This directory contains the implementation of a market game based on the Dubey model. The Dubey model is a theoretical framework for analyzing trading strategies and market dynamics among traders with different endowments and utility functions. More concrete definitions can be found in the no_regret directory, however here is the initial implementations of the Dubey model, a strategy, and a player.

### `nash/`
This directory contains the implementation of a Nash Equilibrium algorithm. It is still wip, and is not yet fully working. The algorithm attempts to find a Nash Equilibrium for a given game, and is based on Robert Wilson's algorithm for finding a Nash Equilibrium.

### `no_regret/`
This directory is dedicated to the implementation and analysis of no-regret learning algorithms, with a particular emphasis on Joint Policy-Space Response Oracles (JPSRO). It includes simple multiplicative weights algorithms and tests these learner's ability to approximate CCE using low-dimensionality games. It also uses JPSRO algorithms to estimate CCE within a custom bid-based trading environment (Dubey's game), leveraging TensorFlow for underlying computations. This directory also provides essential utilities and test scripts to visualize the outputs of these training runs.
