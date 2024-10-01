// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

import "prb-math/contracts/PRBMathSD59x18.sol";


struct platformRight {
    bytes4 platformId;
    uint16 tradePeriod;
    uint16 tradePrice;
    int16 tradeAmount;
}
//
// Colored coin is represented by these 2 contracts
//
// closer to the implementation of an NFT (colored coin is a bitcoin architecture concept primarily)
// for a more detailed approach to implement unique tokens, look up the specification for ERC-721,
// the specification for NFTs (which is also just a smart contract with specific functions)
//
// The coin belongs to a specific user, and the coin is intended to be used as verification 
// for membership on a certain platform or contract, authenticated by being passed to the central infrastructure 
// when purchasing rights for a new platform or contract or entering one that is already purchased

contract PlatformCoin {
    
    // properties of the coin
    //
    // centralVerifier: the address of the account trusted to mint and verify the coin
    // userAccount: the address of the account entitled to the platforms on this coin
    // platforms: the list of IDs of platforms on this coin, each platform is a 4 byte ID

    address public centralVerifier;
    address public userAccount;
    platformRight[] private platforms;

    /* 
    * This function creates the coin, is called automatically on creation
    * takes in the coin's user address and sets the trusted verifier to the 
    * address of the creator
    */
    constructor(address user) payable{
        centralVerifier = msg.sender;
        userAccount = user;
    }

    /*
    * This function checks if a user is already a member of a platform
    * returns true if the caller is a member of > 0 platforms, false otherwise
    */
    function isMemberOfAny() public view returns (bool) {
        if(platforms.length > 0) return true;
        return false;
    }

    /*
    * Checks if a user can add a platform without violating rules of mutual exclusion
    * Returns true if platform can be added, false otherwise
    */
    function canAddPlatform() public view returns (bool){
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        return true;
        // if(isMemberOfAny()){
        //     return false;
        // }
        // return true;
    }

    /*
    * This function adds the given platform ID to the list of platforms on the coin
    * Requires the caller to be the trusted verifier
    */
    function addPlatform(bytes4 platformID, uint16 period, uint16 price, int16 amount) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        platforms.push(platformRight(platformID, period, price, amount));
    }

    /*
    * This function adds the given platform ID to the list of platforms on the coin
    * Requires the caller to be the trusted verifier
    * Rejects the transaction if the user is a member of another platform
    */
    function safeAddPlatform(bytes4 platformID, uint16 period, uint16 price, int16 amount) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        require(canAddPlatform(), "Rejected. User is already a member of a different platform");
        platforms.push(platformRight(platformID, period, price, amount));
    }

    /*
    * Returns the number of platforms the coin has listed
    */
    function getNumPlatforms() public view returns (int){
        return (int)(platforms.length);
    }

    /*
    * Returns the complete list of platform IDs the coin has listed
    * Requires the caller to be the trusted verifier
    */
    function getAllPlatforms() public view returns (platformRight[] memory platforms_){
        require(msg.sender == centralVerifier, "Unathorized account requesting platform list");
        platforms_ = platforms;
    }

    /*
    * Debugging function used by Sam during programming. Returns 
    * bytes data encodings of strings to test parts of the code
    */
    function toString() public pure returns (bytes32){
        string memory myString = "Central verifier";
        bytes32 stringBytes;
        // Convert string to bytes32
        assembly {
            stringBytes := mload(add(myString, 32))
        }
        return stringBytes;
    }
}

contract ContractCoin {
    
    // properties of the coin
    //
    // centralVerifier: the address of the account trusted to mint and verify the coin
    // userAccount: the address of the account entitled to the contracts on this coin
    // contracts: the list of IDs of contracts on this coin, each contracts is a 4 byte ID

    address public centralVerifier;
    address public userAccount;
    bytes4[] private contracts;

    /* 
    * This function creates the coin, is called automatically on creation
    * takes in the coin's user address and sets the trusted verifier to the 
    * address of the creator
    */
    constructor(address user) payable{
        centralVerifier = msg.sender;
        userAccount = user;
    }


    function contractConflict(bytes4 x, bytes4 y) private pure returns (bool){
        if(x == y) return true;
        return false;
    }

    /*
    * Checks if a contract passed in can be added without causing any economic
    * conflicts with the already existing contracts on the coin
    * Returns true if adding causes no conflicts, false otherwise
    */
    function canAddContract(bytes4 contractID) public view returns (bool){
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        for(uint i = 0; i < contracts.length; i++){
            if(contractConflict(contractID, contracts[i])){
                return false;
            }
        }
        return true;
    }

    /*
    * This function adds the given contract ID to the list of contracts on the coin
    * Requires the caller to be the trusted verifier
    */
    function addContract(bytes4 contractID) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        contracts.push(contractID);
    }

    /*
    * This function adds the given contract ID to the list of contracts on the coin
    * Requires the caller to be the trusted verifier
    */
    function safeAddContract(bytes4 contractID) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        require(canAddContract(contractID), "Rejected. Contract conflicts with previously entered contract");
        contracts.push(contractID);
    }

    /*
    * Returns the number of contracts the coin has listed
    */
    function getNumContracts() public view returns (int){
        return (int)(contracts.length);
    }

    /*
    * Returns the complete list of contract IDs the coin has listed
    * Requires the caller to be the trusted verifier
    */
    function getAllContracts() public view returns (bytes4[] memory contracts_){
        require(msg.sender == centralVerifier, "Unathorized account requesting contract list");
        contracts_ = contracts;
    }

    /*
    * Debugging function used by Sam during programming. Returns 
    * bytes data encodings of strings to test parts of the code
    */
    function toString() public pure returns (bytes32){
        string memory myString = "Central verifier";
        bytes32 stringBytes;
        // Convert string to bytes32
        assembly {
            stringBytes := mload(add(myString, 32))
        }
        return stringBytes;
    }
}

//
// This contract is a representation of a set (not a native data type in solidity)
// Lets you add elements to a set and check membership efficiently
//
contract AddressSet {
    // Keeps track of the elements by mapping them to true/false values
    mapping(address => bool) private set;

    /*
    * Adds the passed in address to the set
    */
    function add(address element) public payable {
        set[element] = true;
    }

    /*
    * Returns true if the passed in element is in the set, 
    * false otherwise
    */
    function contains(address element) public view returns (bool) {
        return set[element];
    }
}

//
// This contract is a representation of a set (not a native data type in solidity)
// Lets you add elements to a set and check membership efficiently
//
contract Bytes4Set {
    // Keeps track of the elements by mapping them to true/false values
    mapping(bytes4 => bool) private set;

    constructor(bytes4[] memory initialSet) {
        for(uint i = 0; i < initialSet.length; i++){
            set[initialSet[i]] = true;
        }
    }

    /*
    * Adds the passed in address to the set
    */
    function add(bytes4 element) public payable {
        set[element] = true;
    }

    /*
    * Returns true if the passed in element is in the set, 
    * false otherwise
    */
    function contains(bytes4 element) public view returns (bool) {
        return set[element];
    }
}

//
// This contract controls what platforms/contracts a user has/doesn't have access to
//
// The contract is designed to allocate "colored coins" to users, then 
// verify and modify those coins to register membership of different platforms/contracts
//
contract MultiCoinController {
    
    // properties
    //
    // centralVerifier: the address of the account trusted to determine platform membership
    // allPlatforms: the list of platform IDs for all platforms available
    // allocatedUsers: the set of all users with a coin, in order to prevent one user from having 
    //                 multiple coins

    address public centralVerifier;
    Bytes4Set private allPlatforms;
    AddressSet private allocatedUsers;
    mapping(address => address) public userCoins;
    uint16[] private spotPrices;


    /* 
    * Creates the coin controller contract, assumes the caller of the contract is the trusted 
    * verifier, and takes as input the list of all platform IDs
    */
    constructor(bytes4[] memory platformList) payable {
        centralVerifier = msg.sender;
        allPlatforms = new Bytes4Set(platformList);
        allocatedUsers = new AddressSet();
    }

    /*
    * This function creates a new coin and returns the address 
    * It is intended to be called by a user who has not yet been allocated a coin, and the 
    * created coin is made specifically for the calling user
    * Throws an error if the user already has a coin created
    */
    function getPlatformCoin() public payable returns (address){
        require(!allocatedUsers.contains(msg.sender), "User already has an allocated platform Colored Coin");
        allocatedUsers.add(msg.sender);
        address newCoinAddress = address(new PlatformCoin(msg.sender));
        userCoins[msg.sender] = newCoinAddress;
        return newCoinAddress;
    }

    /*
    * Returns the address to the platform coin of the user calling the function
    * requires the user to have already been allocated a coin, throws an error otherwise
    */
    function getUserCoin() public view returns (address) {
        require(allocatedUsers.contains(msg.sender), "User does not have an allocated platform Colored Coin");
        return userCoins[msg.sender];
    }

    /*
    * given an array of acceptable spot prices, sets the list of acceptable prices to the given list
    */
    function setSpotPrices(uint16[] memory prices) public payable {
        spotPrices = prices;
    }

    /*
    * This function checks if a user is already a member of a platform
    * it takes as input the colored coin of the function caller, and returns true if 
    * the caller is a member of > 0 platforms, false otherwise
    * Throws an error if the colored coin doesn't belong to the caller
    */
    function isMemberOfAnyPlatform(PlatformCoin userCoin) public view returns (bool) {
        require(userCoin.userAccount() == msg.sender, "Colored Coin does not belong to the request user");
        return userCoin.isMemberOfAny();
    }

    /*
    * This function takes as input a colored coin and a platform ID and 
    * adds that platform to the list of platforms belonging to the coin IFF
    * the user is not a member of any other platform and the coin belongs to the 
    * caller
    */
    function addPlatform(PlatformCoin userCoin, bytes4 platformID, uint16 period, uint16 price, int16 amount) public payable {
        require(allPlatforms.contains(platformID), "Platform ID is not a valid ID");
        require(userCoin.userAccount() == msg.sender, "Colored Coin does not belong to the given user");
        bool acceptedPrice = false;
        for(uint i = 0; i < spotPrices.length; i++){
            if(spotPrices[i] == price){
                acceptedPrice = true;
            }
        }
        require(acceptedPrice, "Given price is not an accepted price");
        userCoin.safeAddPlatform(platformID, period, price, amount);
    }

    /*
    * Given a platform coin, returns arrays containing the data of all platforms belonging to that coin
    * Returns arrays for platform IDs, time periods of the platform entitlement, prices allowed to trade at, and amounts allowed to trade, in that order
    * 
    * All arrays are separate, but indexed identially so each element i of all 4 arrays corresponds to the specification of one 
    * platform entitlement
    */
    function listCoinPlatforms(PlatformCoin userCoin) public view returns(bytes4[] memory _platformIds, uint16[] memory _periods, uint16[] memory _prices, int16[] memory _amounts){
        require(msg.sender == centralVerifier, "Request is not authorized: you must be the platform issuer to access this data");

        platformRight[] memory platforms = userCoin.getAllPlatforms();
        bytes4[] memory platformIds = new bytes4[](platforms.length);
        uint16[] memory prices = new uint16[](platforms.length);
        uint16[] memory periods = new uint16[](platforms.length);
        int16[] memory amounts = new int16[](platforms.length);
        for(uint i = 0; i < platforms.length; i++){
            platformIds[i] = platforms[i].platformId;
            prices[i] = platforms[i].tradePrice;
            periods[i] = platforms[i].tradePeriod;
            amounts[i] = platforms[i].tradeAmount;
        }
        _platformIds = platformIds;
        _periods = periods;
        _prices = prices;
        _amounts = amounts;
    }
}