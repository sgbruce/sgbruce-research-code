import Web3 from 'web3'
import { Contract, ContractSendMethod, Options } from 'web3-eth-contract'
import { deploy, getAbi } from './web3-lib'
import * as data from '../data/JPEex';

const web3 = new Web3(web3Provider);

type PlatformRight = {
  platformId: string;
  tradePrice: number;
  tradePeriod: number;
}

const runExample = async () => {
  try {
    const accounts = await web3.eth.getAccounts();

    // Keep track of the deployed contract status
    const deployments: Promise<any>[] = []; 

    // deploy the platform controller with list of valid platform IDs 
    deployments.push(deploy("MultiCoinController", [["0x00000001", "0x00000002", "0x00000003"]]));

    await Promise.all(deployments);
    console.log(`All contracts deployed`);
    // Create a contract object with the deployed controller just created
    const controller: Contract = new web3.eth.Contract(await getAbi("MultiCoinController"), (await deployments[0]).address);

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

    // Create and get the platform coin belonging to the account in this file
    // First calls a transaction on the controller to create a coin, then calls a get function 
    // to recieve the address of the coin, returns a promise to the string address of the coin
    const getPlatformCoin = async (): Promise<string> => {
      await controller.methods.getPlatformCoin().send({from: accounts[0], gas: 1500000}).on("error", console.error);
      let resCoin = "nan";
      await controller.methods.getUserCoin().call((error, result) => {
            if (!error) {
                console.log(`Current coin: ${result}`);
                resCoin = result;
            } else {
                console.error('Error:', error);
            }
        });
        return resCoin;
    }

    // Get a platform coin for the user
    const userCoin = await getPlatformCoin();
    // Create a contract object with the coin just created
    const coin: Contract = new web3.eth.Contract( await getAbi("PlatformCoin"), userCoin);

    // Get current number of platform entitlements on the coin, should be 0
    await getContractProp(coin.methods.getNumPlatforms, "number of platforms");

    // Set the acceptable spot prices allowed for trading in future time periods
    await executeTransaction(controller.methods.setSpotPrices, "setting spot prices", [[5770, 5974, 6181]]);

    // Add two platforms as entitlements for time period 2, at the prices and quantities described in the JPE paper
    // Prices and amounts are multiplied by 10,000 to allow integer math in the smart contract
    await executeTransaction(controller.methods.addPlatform, "adding platform", [coin.options.address, "0x00000001", 2, 5974, 2970]);
    await executeTransaction(controller.methods.addPlatform, "adding platform", [coin.options.address, "0x00000002", 2, 5974, -2970]);
    // Get the number of platforms now on the coin, should be 2
    await getContractProp(coin.methods.getNumPlatforms, "number of platforms");

    // Get the in depth information of each contract on the coin using a contract function listCoinPlatforms
    console.log(`Fetching platform info`);
    await controller.methods.listCoinPlatforms(coin.options.address).call((error, result) => {
        if (!error) {
            console.log(`Current platform IDs: ${result[0]}`);
            console.log(`Current trade Periods: ${result[1]}`);
            console.log(`Current trade Prices: ${result[2].map((el) => el/10000)}`);
            console.log(`Current trade Amounts: ${result[3].map((el) => el/10000)}`);
        } else {
            console.error('Error:', error);
        }
    });

  } catch (e) {
    console.log(e.message)
  }
};

runExample();