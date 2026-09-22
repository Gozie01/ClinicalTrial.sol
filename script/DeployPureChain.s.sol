// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../contracts/TrustCTGovernance.sol";

interface VmBroadcast {
    function envUint(string calldata) external returns (uint256);
    function envAddress(string calldata) external returns (address);
    function startBroadcast(uint256 privateKey) external;
    function stopBroadcast() external;
}

contract DeployPureChainScript {
    VmBroadcast internal constant vm = VmBroadcast(address(uint160(uint256(keccak256("hevm cheat code")))));

    function run() external returns (TrustCTGovernance deployed) {
        uint256 privateKey = vm.envUint("PURECHAIN_PRIVATE_KEY");
        address trialManager = vm.envAddress("PURECHAIN_TRIAL_MANAGER");

        address[] memory validators = new address[](3);
        validators[0] = vm.envAddress("PURECHAIN_VALIDATOR_A");
        validators[1] = vm.envAddress("PURECHAIN_VALIDATOR_B");
        validators[2] = vm.envAddress("PURECHAIN_VALIDATOR_C");

        vm.startBroadcast(privateKey);
        deployed = new TrustCTGovernance(trialManager, validators, 2, 3);
        vm.stopBroadcast();
    }
}
