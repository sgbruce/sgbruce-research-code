// SPDX-License-Identifier: MIT
pragma solidity >=0.6.12 <0.9.0;

// INITIAL PASS - 2 PLAYERS

uint constant PROB_MAX_INT = 10000;

contract CorrelatedPlanner {

    struct ProfileEntry {
        address player;
        string strategy;
    }

    // Keep track of strategies, both as integers for optimized computing and as strings for human readability
    mapping(string => uint) private strategy_map;
    string[] private strategy_list;
    uint[] private strategies;

    // Keep track of players, both as addresses for identification and as integers for optimized computing
    mapping(address => uint) private player_map;
    address[] private player_list;
    uint[] private players;

    // Keep track of utilities for each player and strategy combination
    int[][][] private utilities;

    // Keep track of the probability distribution over strategy combinations
    uint[] private probabilities;
    uint[][] private combinations;

    // Initialize a contract with the set of strategies
    constructor(string[] memory _strategies){ // Estimated creation cost: infinite gas Estimated code deposit cost: 1289400 gas
        for(uint i = 0; i < _strategies.length; i++){
            strategy_map[_strategies[i]] = i;
            strategies.push(i);
        }
        strategy_list = _strategies;
        player_list = new address[](0);
        players = new uint[](0);
        utilities = new int[][][](0);
        probabilities = new uint[](0);
        combinations = new uint[][](0);
    }

    // Enumerate all possible strategy combinations over the set of players
    function enumerate_strategy_combinations() internal view returns (uint[][] memory) { // Estimated execution cost: infinite gas
        uint num_players = players.length;
        uint num_strategies = strategies.length;
        uint total_combinations = 1;
        for (uint i = 0; i < num_players; i++) {
            total_combinations *= num_strategies;
        }

        uint[][] memory new_combinations = new uint[][](total_combinations);
        for (uint i = 0; i < total_combinations; i++) {
            new_combinations[i] = new uint[](num_players);
            uint temp = i;
            for (uint j = 0; j < num_players; j++) {
                new_combinations[i][j] = temp % num_strategies;
                temp /= num_strategies;
            }
        }
        return new_combinations;
    }

    // Get all strategy combinations, for testing purposes
    function get_all_strategy_profiles() public view returns (uint[][] memory) { // Estimated execution cost: infinite gas
        uint[][] memory all_combinations = enumerate_strategy_combinations();
        return all_combinations;
    }

    // Map a list of strategy indices to a list of profiles, for human readability and testing purposes
    function map_list_to_profile(uint[] memory profile_list) internal view returns (ProfileEntry[] memory) { // Estimated execution cost: undefined gas
        ProfileEntry[] memory profile = new ProfileEntry[](profile_list.length);
        for (uint i = 0; i < profile_list.length; i++) {
            profile[i] = ProfileEntry(player_list[i], strategy_list[profile_list[i]]);
        }
        return profile;
    }

    // Build the constraints for the linear program, one for each player's alternate strategy to a signalled strategy
    function build_ic_constraints() internal view returns (int[][] memory, int[] memory) { // Estimated execution cost: infinite gas
        uint num_constraints = players.length * strategies.length * (strategies.length - 1);
        uint num_variables = combinations.length;
        int[][] memory A_ub = new int[][](num_constraints);
        for (uint i = 0; i < num_constraints; i++) {
            A_ub[i] = new int[](num_variables);
        }
        int[] memory b_ub = new int[](num_constraints);

        for (uint index = 0; index < num_variables; index++) {
            for (uint player = 0; player < players.length; player++) {
                uint[] memory profile = new uint[](combinations[index].length);
                for (uint i = 0; i < profile.length; i++) {
                    profile[i] = combinations[index][i];
                }
                int player_utility = utilities[player][profile[0]][profile[1]]; // DOES NOT WORK FOR MORE THAN 2 PLAYERS
                uint strategy = profile[player];
                for (uint alternate_strategy = 0; alternate_strategy < strategies.length; alternate_strategy++) {
                    if (alternate_strategy != strategy) {
                        profile[player] = alternate_strategy;
                        int deviation_utility = utilities[player][profile[0]][profile[1]]; // DOES NOT WORK FOR MORE THAN 2 PLAYERS
                        uint alt_index = alternate_strategy > strategy ? alternate_strategy - 1 : alternate_strategy;
                        uint constraint_index = player * strategies.length * (strategies.length - 1) + strategy * (strategies.length - 1) + alt_index;
                        A_ub[constraint_index][index] = deviation_utility - player_utility;
                    }
                }
            }
        }
        return (A_ub, b_ub);
    }

    // Initialize the distribution with equal probability for each strategy combination
    function initialize_distribution() internal { // Estimated execution cost: infinite gas
        uint[][] memory all_combinations = enumerate_strategy_combinations();
        probabilities = new uint[](all_combinations.length);
        combinations = new uint[][](all_combinations.length);
        for (uint i = 0; i < all_combinations.length; i++) {
            probabilities[i] = PROB_MAX_INT / all_combinations.length;
            combinations[i] = all_combinations[i];
        }
    }

    // Get the lambdas for the linear program, currently set to equal weights
    function get_lambdas() internal view returns (int[] memory) { // Estimated execution cost: infinite gas
        int[] memory lambdas = new int[](players.length);
        for(uint i = 0; i < lambdas.length; i++){
            lambdas[i] = 1;
        }
        return lambdas;
    }

    // Optimize the distribution using linear programming
    function optimize_distribution() public payable returns (int[] memory) { // Estimated execution cost: infinite gas
        if(combinations.length == 0) {
            initialize_distribution();
        }
        int[] memory lambdas = get_lambdas();
        int[] memory neg_utility_sums = new int[](combinations.length);
        for (uint i = 0; i < combinations.length; i++) {
            for (uint k = 0; k < players.length; k++) {
                neg_utility_sums[i] -= lambdas[k] * utilities[k][combinations[i][0]][combinations[i][1]]; // DOES NOT WORK FOR MORE THAN 2 PLAYERS
            }
        }
        (int[][] memory A_ub, int[] memory b_ub) = build_ic_constraints();
        int[] memory b_eq = new int[](1);
        b_eq[0] = 1;
        int[][] memory A_eq = new int[][](1);
        A_eq[0] = new int[](combinations.length);
        for(uint i = 0; i < combinations.length; i++){
            A_eq[0][i] = 1;
        }
        // Call the linear programming function here, using c = neg_utility_sums, A_ub, b_ub, A_eq, b_eq
        // Set the result to the distribution
        return neg_utility_sums;
    }
    
    // Add a player to the contract and recaculate the correlated equilibrium
    // Assume an anonymous game where each player's utility is only dependent on their own strategy
    // and the aggregate strategy of the other players
    function add_player(address player, int[][] memory _utilities) public payable { // Estimated execution cost: infinite gas
        require(_utilities.length == strategies.length, "Utilities must be the same length as the number of strategies");
        require(players.length  < 2, "Only 2 players are supported in this contract");
        uint player_id = players.length;
        player_map[player] = player_id;
        player_list.push(player);
        players.push(player_id);
        utilities.push(_utilities);
    }
}