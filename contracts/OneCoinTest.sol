// SPDX-License-Identifier: MIT
pragma solidity >=0.8.0 <0.9.0;

//
// Colored coin is represented by this contract
//
// closer to the implementation of an NFT (colored coin is a bitcoin architecture concept primarily)
// for a more detailed approach to implement unique tokens, look up the specification for ERC-721,
// the specification for NFTs (which is also just a smart contract with specific functions)
//
// The coin belongs to a specific user, and the coin is intended to be used as verification 
// for membership on a certain platform, authenticated by being passed to the central infrastructure 
// when purchasing rights for a new platform or entering one that is already purchased

contract ColoredCoin {
    
    // properties of the coin
    //
    // centralVerifier: the address of the account trusted to mint and verify the coin
    // userAccount: the address of the account entitled to the platforms/contracts on this coin
    // commitments: the list of IDs of platforms and contracts on this coin, each ID is a 4 byte number

    address public centralVerifier;
    address public userAccount;
    bytes4[] private commitments;

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
    * Checks if there is a conflict between two contracts x and y
    * conflicts refer to economic constraints, i.e. multi-principal agent prolems
    * or scenarios where mutual exclusion is necessary
    * Returns true if there is a conflict, false otherwise
    */
    function commitmentConflict(bytes4 x, bytes4 y) private pure returns (bool){
        bytes1 platformByte = 0x80;
        if(x[x.length - 1] == platformByte && y[y.length - 1] == platformByte){
            return true;
        }
        return false;
    }

    /*
    * Checks if a commitment passed in can be added without causing any economic
    * conflicts with the already existing commitments on the coin
    * Returns true if adding causes no conflicts, false otherwise
    */
    function canAddCommitment(bytes4 ID) public view returns (bool){
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        for(uint i = 0; i < commitments.length; i++){
            if(commitmentConflict(commitments[i], ID)){
                return false;
            }
        }
        return true;
    }

    /*
    * This function adds the given commitment ID to the list of commitments on the coin
    * Requires the caller to be the trusted verifier
    */
    function addCommitment(bytes4 ID) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        commitments.push(ID);
    }

    /*
    * This function adds the given commitment ID to the list of commitments on the coin
    * and verifies that commitment is valid to add before adding
    * Requires the caller to be the trusted verifier
    */
    function safeAddCommitment(bytes4 ID) public payable{
        require(msg.sender == centralVerifier, "Permission denied. Requestor is not the central verifier");
        require(canAddCommitment(ID), "Commitments cannot be added, conflicts with previous commitment");
        commitments.push(ID);
    }

    /*
    * Returns the number of commitments the coin has listed
    */
    function getNumCommitments() public view returns (int){
        return (int)(commitments.length);
    }

    /*
    * Returns true if user is a member of a platform, false otherwise
    */
    function isMemberOfPlatform() public view returns (bool){
        bytes1 platformByte = 0x80;
        for(uint i = 0; i < commitments.length; i++){
            if(commitments[i] == platformByte) {
                return true;
            }
        }
        return false;
    }

    /*
    * Returns the complete list of platform IDs the coin has listed
    * Requires the caller to be the trusted verifier
    */
    function getAllCommitments() public view returns (bytes4[] memory _commitments){
        require(msg.sender == centralVerifier, "Unathorized account requesting commitment list");
        _commitments = commitments;
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
// This contract controls what platforms a user has/doesn't have access to
//
// The contract is designed to allocate "colored coins" to users, then 
// verify and modify those coins to register membership of different platforms
//
contract OneCoinController {
    
    // properties
    //
    // centralVerifier: the address of the account trusted to determine platform membership
    // allPlatforms: the list of platform IDs for all platforms available
    // allocatedUsers: the set of all users with a coin, in order to prevent one user from having 
    //                 multiple coins

    address public centralVerifier;
    Bytes4Set private allPlatforms;
    AddressSet private allocatedUsers;

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
    function getCoin() public payable returns (address){
        require(!allocatedUsers.contains(msg.sender), "User already has an allocated Colored Coin");
        allocatedUsers.add(msg.sender);
        return address(new ColoredCoin(msg.sender));
    }

    /*
    * This function checks if a user is already a member of a platform
    * it takes as input the colored coin of the function caller, and returns true if 
    * the caller is a member of > 0 platforms, false otherwise
    * Throws an error if the colored coin doesn't belong to the caller
    */
    function isMemberOfPlatform(ColoredCoin userCoin) public view returns (bool) {
        require(userCoin.userAccount() == msg.sender, "Colored Coin does not belong to the request user");
        return userCoin.isMemberOfPlatform();
    }

    /*
    * This function takes as input a colored coin and a platform ID and 
    * adds that platform to the list of platforms belonging to the coin IFF
    * the user is not a member of any other platform and the coin belongs to the 
    * caller
    */
    function addCommitment(ColoredCoin userCoin, bytes4 ID) public payable {
        if(ID[0] == 0x80){
            require(allPlatforms.contains(ID), "Platform ID is not a valid ID");
        }
        require(userCoin.userAccount() == msg.sender, "Colored Coin does not belong to the given user");
        userCoin.safeAddCommitment(ID);
    }

    function listCoinCommitments(ColoredCoin userCoin) public view returns(bytes4[] memory _ret){
        require(msg.sender == centralVerifier, "Request is not authorized: you must be the platform issuer to access this data");
        _ret = userCoin.getAllCommitments();
    }
}