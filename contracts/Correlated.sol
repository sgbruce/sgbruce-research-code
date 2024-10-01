// SPDX-License-Identifier: MIT
pragma solidity >=0.6.12 <0.9.0;

contract CorrelatedPlanner {
    int[] private strategies;
    address[] private players;
    mapping(address => int[]) private utilities;

    // Initialize a contract with the pricipal, agent, and given outcomes
    constructor(int[] memory strategies){
        strategies = strategies;
        players = [];
    }

    function get_correlated_equilibrium() private {
        
    }

    // Add a player to the contract and recaculate the correlated equilibrium
    // Assume an anonymous game where each player's utility is only dependent on their own strategy
    // and the aggregate strategy of the other players
    function add_player(address player, int[] memory utilities) public payable {
        require(utilities.length == strategies.length, "Utilities must be the same length as the number of strategies");
        players.push(player);
        utilities[player] = utilities;
    }
}