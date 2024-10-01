// SPDX-License-Identifier: MIT
pragma solidity >=0.6.12 <0.9.0;

/***
* This contract represents a contract offered by a principal to an agent.
* The contract stipulates payouts to an agent "agent" from a principal "principal" 
* and vice versa based on specific outcomes, represented as strings. 
*
* Required standard functions
*
* constructor: takes as input the principal address, the agent address, and a list of {outcome, qi, yi} sets
* qi(outcome): takes an outcome as input and returns the state contingent payout to the agent 
* yi(outcome): takes an outcome as input and returns the state contingent payout to the principal
* getAllOutcomes(): returns a list of all outcomes where there is a non-zero payoff to principal or agent
***/
contract Contract {
    // Addresses of the pricipal and agent accounts
    address public principal;
    address public agent;

    // qi maps outcome strings to agent payoffs as integer amounts, 0 for any outcome not stipulated in the contract
    mapping(string => int) public qi;
    // yi maps outcome strings to principal payoffs as integer amounts, 0 for any outcome not stipulated in the contract
    mapping(string => int) public yi;
    // lists all non-zero outcomes in the contract
    string[] private outcomes;

    // Structure for defining outcome inputs, contains an outcome string, and agent and principal payoffs
    struct contractOutcomeType{
        string outcome;
        int principalPayoff;
        int agentPayout;
    }

    // Initialize a contract with the pricipal, agent, and given outcomes
    constructor(address p, address a, contractOutcomeType[] memory contractOutputs){
        principal = p;
        agent = a; 
        // For each given outcome, add the mapping of outcome to payoffs and record the non-zero outcome 
        for(uint i = 0; i < contractOutputs.length; i++){
            string memory o = contractOutputs[i].outcome;
            qi[o] = contractOutputs[i].principalPayoff;
            yi[o] = contractOutputs[i].agentPayout;
            outcomes.push(o);
        }
    }

    function getAllOutcomes() public view returns (string[] memory){
        return outcomes;
    }
}

/***
* This contract manages the set of all contracts 
* When a new contract is added, updates the state contingent payouts offered to the agents 
* by the contract manager to ensure an optimal equilibrium
***/
contract ContractManager {
    // Payment schemes to the principals of the form B_i - ( J - 1/J)y* + (q - q^i)
    // B_i is vector that guarantees net payoff is Q_i for each principle
    //  Q_i is the principal payoff if agent does not participate
    // J = num principals 
    // y^i* = optimal payments from principal i to agent (vector over outcomes pi)
    // y* = sum of all y^i*
    // q^i = payoff to principal i from outcome (vector over outcomes pi)
    // q = sum of all q^i

    // Keeps track of principals offering contracts
    mapping(address => bool) principals;
    // List of all offered contracts, is this needed?
    Contract[] private contracts;
    // Number of participating principals
    int public numPrincipals = 0;
    // Payoff to principals for non-participation
    int public noParticipationPayoff = 0;

    // Total state contingent payoffs across all contracts based on outcome
    mapping(string => int) public totalY;
    mapping(string => int) public totalQ;
    
    // Payouts to principals offered from contract manager, based on the last outcome assessed
    // in this contract
    // accessible in the form "principalPayouts(principal address)"
    mapping(address => int) public principalPayouts;

    // Given a contract, adds all non-zero principal payoffs to the aggregate based on outcome
    function addYi(Contract c) private{
        string[] memory contractOutcomes = c.getAllOutcomes();
        for(uint256 i = 0; i < contractOutcomes.length; i++) {
            totalY[contractOutcomes[i]] += c.yi(contractOutcomes[i]);
        }
    }

    // Given a contract, adds all non-zero agent payoffs to the aggregate based on outcome
    function addQi(Contract c) private{
        string[] memory contractOutcomes = c.getAllOutcomes();
        for(uint256 i = 0; i < contractOutcomes.length; i++) {
            totalQ[contractOutcomes[i]] += c.qi(contractOutcomes[i]);
        }
    }

    // Given a new contract, updates the aggregate agent and principal payouts and stores record of the contract
    // For now requires each principal to only offer 1 contract, how can we fix this?
    function addContract(Contract c) public payable{
        require(principals[c.principal()] == false, "Principal has already entered in a contract");
        principals[c.principal()] = true;
        numPrincipals++;
        addQi(c);
        addYi(c);
        contracts.push(c);
    } 

    // Given an outcome as a string, updates the payouts to the principals per the contract with the contract manager, and stores 
    // in the principalPayouts data structure 
    function assessOutcome(string memory outcome)public payable{
        for(uint256 i = 0; i < contracts.length; i++){
            // Payment schemes to the principals of the form B_i - ( J - 1/J)y* + (q - q^i)
            principalPayouts[contracts[i].principal()] = 0;
        }
        for(uint256 i = 0; i < contracts.length; i++){
            // Payment schemes to the principals of the form B_i - ( J - 1/J)y* + (q - q^i)
            //principalPayouts[contracts[i].principal()] += 10;
            principalPayouts[contracts[i].principal()] += -(numPrincipals - (1/numPrincipals)) * totalY[outcome] + (totalQ[outcome] - contracts[i].qi(outcome));
        }
    }

    function getNumContracts() public view returns(uint) {
        return contracts.length;
    }
}
