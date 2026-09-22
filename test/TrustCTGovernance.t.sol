// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../contracts/TrustCTGovernance.sol";

interface Vm {
    function prank(address) external;
    function startPrank(address) external;
    function stopPrank() external;
    function roll(uint256) external;
    function expectRevert(bytes4) external;
}

contract TrustCTGovernanceTest {
    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    TrustCTGovernance internal governance;

    address internal manager = address(0x100);
    address internal validatorA = address(0x201);
    address internal validatorB = address(0x202);
    address internal validatorC = address(0x203);
    address internal participantA = address(0x301);
    address internal participantB = address(0x302);

    function setUp() public {
        address[] memory validators = new address[](3);
        validators[0] = validatorA;
        validators[1] = validatorB;
        validators[2] = validatorC;

        governance = new TrustCTGovernance(manager, validators, 2, 3);
        governance.registerParticipant(participantA, true);
        governance.registerParticipant(participantB, true);
    }

    function test_ReplayAttendanceBlocked() public {
        vm.prank(manager);
        governance.logAttendance(participantA, 1, 1, 77);

        vm.prank(manager);
        vm.expectRevert(TrustCTGovernance.ReplayDetected.selector);
        governance.logAttendance(participantA, 1, 1, 77);
    }

    function test_MaliciousParticipantCannotForgeAttendanceForAnotherParticipant() public {
        vm.prank(participantA);
        vm.expectRevert(TrustCTGovernance.Unauthorized.selector);
        governance.logAttendance(participantB, 1, 5, 11);
    }

    function test_CannotExecuteBeforeFinality() public {
        uint256 claimId = _prepareCanonicalClaim(participantA, 1, 2, 125);

        vm.expectRevert(TrustCTGovernance.FinalityPending.selector);
        governance.executeReward(claimId);
    }

    function test_OnlyValidatorsCanApprove() public {
        vm.prank(manager);
        governance.logAttendance(participantA, 1, 9, 17);
        vm.prank(manager);
        uint256 claimId = governance.proposeReward(participantA, 1, 9, 100, 18);

        vm.prank(participantA);
        vm.expectRevert(TrustCTGovernance.Unauthorized.selector);
        governance.approveReward(claimId);
    }

    function test_ConflictingRewardExecutionBlocked() public {
        vm.prank(manager);
        governance.logAttendance(participantA, 1, 4, 21);

        vm.prank(manager);
        uint256 claimA = governance.proposeReward(participantA, 1, 4, 200, 22);
        vm.prank(manager);
        uint256 claimB = governance.proposeReward(participantA, 1, 4, 350, 23);

        _approveByQuorum(claimA);
        _approveByQuorum(claimB);
        vm.roll(block.number + 4);

        governance.executeReward(claimA);

        vm.expectRevert(TrustCTGovernance.VisitAlreadyRewarded.selector);
        governance.executeReward(claimB);
    }

    function test_CancelledClaimCannotExecute() public {
        vm.prank(manager);
        governance.logAttendance(participantA, 1, 6, 44);
        vm.prank(manager);
        uint256 claimId = governance.proposeReward(participantA, 1, 6, 100, 45);
        _approveByQuorum(claimId);
        governance.cancelClaim(claimId);
        vm.roll(block.number + 4);
        vm.expectRevert(TrustCTGovernance.ClaimCancelled.selector);
        governance.executeReward(claimId);
    }

    function testFuzz_VisitCanOnlyBeRewardedOnce(uint96 rawA, uint96 rawB, uint32 nonceSeed) public {
        uint128 amountA = uint128(uint256(rawA) % 1_000_000 + 1);
        uint128 amountB = uint128(uint256(rawB) % 1_000_000 + 1);
        uint256 nonceA = uint256(nonceSeed) + 100;
        uint256 nonceB = uint256(nonceSeed) + 200;

        vm.prank(manager);
        governance.logAttendance(participantA, 77, 3, nonceA);

        vm.prank(manager);
        uint256 claimA = governance.proposeReward(participantA, 77, 3, amountA, nonceA + 1);
        vm.prank(manager);
        uint256 claimB = governance.proposeReward(participantA, 77, 3, amountB, nonceB + 1);

        _approveByQuorum(claimA);
        _approveByQuorum(claimB);
        vm.roll(block.number + 4);
        governance.executeReward(claimA);

        vm.expectRevert(TrustCTGovernance.VisitAlreadyRewarded.selector);
        governance.executeReward(claimB);
    }

    function _prepareCanonicalClaim(address participant, uint256 trialId, uint256 visitId, uint128 amount) internal returns (uint256 claimId) {
        vm.prank(manager);
        governance.logAttendance(participant, trialId, visitId, 1);
        vm.prank(manager);
        claimId = governance.proposeReward(participant, trialId, visitId, amount, 2);
        _approveByQuorum(claimId);
    }

    function _approveByQuorum(uint256 claimId) internal {
        vm.prank(validatorA);
        governance.approveReward(claimId);
        vm.prank(validatorB);
        governance.approveReward(claimId);
    }
}
