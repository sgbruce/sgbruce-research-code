import Web3 from 'web3'
import { Contract, ContractSendMethod, Options } from 'web3-eth-contract'
import { deploy, getAbi } from './web3-lib'
import { importContracts } from '../data/contractData'

const web3 = new Web3(web3Provider);

const runExample = async () => {
  try {
    const accounts = await web3.eth.getAccounts();

    // Keep track of the deployed contract status
    const deployments: Promise<any>[] = []; 

    // deploy the contract manager
    deployments.push(deploy("ContractManager", []));

    // deploy all of the principal-agent contracts
    let principalNum = 0;
    for(const contract of importContracts){
        deployments.push(deploy("Contract", [accounts[principalNum], accounts[0], contract.outcomes]));
        principalNum++;
    }
    // wait until all contracts have been deployed
    await Promise.all(deployments);
    console.log(`All contracts deployed, there were ${importContracts.length} principal-agent contracts`);
    // Create contract object with the deployed contractManager
    const manager: Contract = new web3.eth.Contract(await getAbi("ContractManager"), (await deployments[0]).address);
    
    // Create list of contract objects with each one representing a deployed contract
    const contracts: Contract[] = [];
    for(const contractPromise of deployments.slice(1)){
        const newContract: Contract = new web3.eth.Contract(await getAbi("Contract"), (await contractPromise).address);
        contracts.push(newContract);
    }
    
    // Calls a function on the contract manager to get the current number of participating principals
    const getNumPrincipals = async () => {
        console.log("Fetching number of principals");
        await manager.methods.numPrincipals().call((error, result) => {
            if (!error) {
                console.log(`Current numPrincipals: ${result}`);
            } else {
                console.error('Error:', error);
            }
        });
    }

    /*
    * Calls a non-payable get function from a contract
    * 
    * Takes as input the contract function to call, the descriptive name of the return value, and optionally 
    * any contract function inputs as an array of inputs
    * Calls the function and prints out the result of the call or any errors thrown 
    */
    const getContractProp = async (func: Function, name: string, input = []) => {
        console.log(`Fetching ${name}`);
        await func(...input).call((error, result) => {
            if (!error) {
                console.log(`Current ${name}: ${result}`);
            } else {
                console.error('Error:', error);
            }
        });
    }

    /*
    * Calls a payable set function from a contract
    *
    * Takes as input the contract function to call, the descriptive name of the action taken, and optionally
    * any contract function inputs as an array of inputs
    * Sends the transaction from the caller's account and 1500000 gas, and prints any errors recieved
    */
    const executeTransaction = async (func: Function, name: string, input = []) => {
      console.log(`executing ${name}`);
        await func(...input).send({from: accounts[0], gas: 1500000}).on("error", console.error);
      console.log(`successfully ${name}`);
    }

    /*
    * Adds a contract to the set of offered contracts in the contract manager
    *
    * Takes as input an address to the contract to be added to the manager (must be an existing smart contract)
    * logs to the console the verification hash from the transaction if suvccessful, error message if not
    */
    const addContract = async (contractAddress: string, debug = false) => {
        console.log(`Adding contract at address ${contractAddress}`);
        await manager.methods.addContract(contractAddress).send({from: accounts[0], gas: 1500000}).on('transactionHash', function(hash){
                                                                                                    debug && console.log('Transaction hash:', hash);
                                                                                                  }).on("error", console.error);
        console.log("Contract added successfully");
    }

    // await getNumPrincipals();
    // await addContract(contracts[0].options.address);
    // await getNumPrincipals();

    // Add all the contracts to the contract manager's set of available contracts
    for(const contract of contracts) {
      await addContract(contract.options.address);
    }

    // Get the total number of contracts offered
    await getContractProp(manager.methods.getNumContracts, `total number of contracts`);

    // Get the total agent and principal payouts for outcome 'high'
    const outcome = 'high';
    await getContractProp(manager.methods.totalY,`total agent payoff for outcome ${outcome}`, [outcome]);
    await getContractProp(manager.methods.totalQ,`total principal payoff for outcome ${outcome}`, [outcome]);
    // Set the outcome to 'high', and get the payout from the principal's principal to the principal
    await executeTransaction(manager.methods.assessOutcome, `assessing outcome ${outcome}`, [outcome]);
    await getContractProp(manager.methods.principalPayouts, `principal payout for ${accounts[0]}`, [accounts[0]]);

    // await getContractProp(contracts[0].methods.principal, 'principal of contract 0');
  
  } catch (e) {
    console.log(e.message)
  }
};

runExample();