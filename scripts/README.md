# Scripts Directory

This directory contains helper code for running and testing the solidity smart contracts in the contracts directory. The files in this directory are in typescript, and handle connectionto blockchain environments as well as deploying contracts to the blockchain, and finally testing said deployed contracts.

## Files

### `deploy_with_ethers.ts`
Simple deploy of a contract to the blockchain using the ethers package.

### `deploy_with_web3.ts`
Simple deploy of a contract to the blockchain using the web3 package.

### `ethers-lib.ts`
This is a slightly more involved example of deployment with the ethers package. It allows specifying the contract name and any inputs at function call time.

### `contractTest.ts`
This file tests the "contract" contract code in the contracts directory. It uses the web3 package to deploy a contract manager as well as several contracts,and offers a testing framework for verifying deployment and proper interaction 

### `jpe.ts`
This testing function operates the same as `contractTest.ts`, except working on the colored coin implementation of the contract code

