# Contracts Directory

This directory is the solidity contract code for blockchain smart contract deployments. It primarily contains constractsaimed at solving the multiple-principal agent problem. The contracts include methods to both explicitly keep track of commitments centrally, as well as methods to allow for the creation of colored coins, where each agent is responsible for tracking their commitments. It also attempts to implement ideas from "Common Ageny" by Bernheim and Winston. 

## Files

### `Contracts.sol`
This contract file is a multi-principal agent environment where commitments are all tracked centrally by a contract manager. This manager is responsible for adding new contracts and for computing payouts to principals to reduce innefficiencies. These payouts are computed using the "principal's principal" defined by Bernheim and Winston's "Common Agency" on page 937 to offer outcome-contingent payments to the principals.

### `OneCoinTest.sol`
This contract file is a colored coin implementation where all commitments are modeled as a single colored coin. There are several contracts specified in the file, including the actual colored coin definition as well as a controller for creation, verification, and modification of the coins.

### `MultiCoinTest.sol`
This contract file is an alternative colored coin implementation with several different types of colored coins that all implement the same/similar functions. The separation here is between contract commitments and platform rights, specified by contractCoins and platformCoins respectively. There is also a controller contract that is used to manage the creation, verification, and modification of the coins as in the OneCoinTest contract.

### `HelloWorld.sol`
This contract file is a simple contract that prints "Hello World!" when the `print` function is called. It is used as a simple example for debugging and deployment validation purposes.

### `Correlated.sol`
This contract attempts to implement the linear programming correlated equilibrium solver from the `correlated` directory in solidity code. The constract is built to add players with utilities over strategies, then explicitly compute equilibria. Currently it does not function fully. The contraint creation and other utilities work for 2 players, however not for >2 and the actualy linear programming solver is missing. This is a good blueprint to use for implementing a CE solver in solidity in the future. 