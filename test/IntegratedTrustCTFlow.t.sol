// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../contracts/CTToken.sol";
import "../contracts/TrialManager.sol";
import "../contracts/IncentiveManager.sol";

interface Vm {
    function prank(address) external;
    function startPrank(address) external;
    function stopPrank() external;
    function roll(uint256) external;
    function expectRevert(bytes4) external;
}

contract IntegratedTrustCTFlowTest {
    Vm internal constant vm = Vm(address(uint160(uint256(keccak256("hevm cheat code")))));

    CTToken internal token;
    TrialManager internal trialManager;
    IncentiveManager internal incentiveManager;

    address internal coordinator = address(0x101);
    address internal validatorA = address(0x201);
    address internal validatorB = address(0x202);
    address internal validatorC = address(0x203);
    address internal participantA = address(0x301);
    address internal participantB = address(0x302);

    string internal attendanceCid = "bafyattendance001";
    string internal modelCid = "bafymodelupdate001";
    string internal incentiveCid = "bafyincentive001";

    function setUp() public {
        address[] memory coordinators = new address[](1);
        coordinators[0] = coordinator;
        trialManager = new TrialManager(coordinators);
        trialManager.registerParticipant(participantA, true);
        trialManager.registerParticipant(participantB, true);

        token = new CTToken(address(this));

        address[] memory validators = new address[](3);
        validators[0] = validatorA;
        validators[1] = validatorB;
        validators[2] = validatorC;
        incentiveManager = new IncentiveManager(address(trialManager), address(token), coordinator, validators, 2, 3);
        token.setMinter(address(incentiveManager));
    }

    function test_ModelArtifactCanBeAnchored() public {
        bytes32 cidHash = keccak256(bytes(modelCid));

        vm.prank(coordinator);
        uint256 artifactId = trialManager.anchorModelUpdate(modelCid, cidHash, keccak256("model-hash"), 1, 11);

        (
            string memory artifactType,
            string memory cid,
            bytes32 storedCidHash,
            bytes32 metadataHash,
            uint64 logicalRef,
            uint64 anchoredBlock,
            address actor
        ) = _readArtifact(artifactId);

        require(keccak256(bytes(artifactType)) == keccak256(bytes("MODEL_UPDATE")), "artifact type mismatch");
        require(keccak256(bytes(cid)) == keccak256(bytes(modelCid)), "cid mismatch");
        require(storedCidHash == cidHash, "cid hash mismatch");
        require(metadataHash == keccak256("model-hash"), "metadata mismatch");
        require(logicalRef == 1, "round mismatch");
        require(anchoredBlock > 0, "anchor block missing");
        require(actor == coordinator, "actor mismatch");
    }

    function test_ReplayAttendanceBlockedWithCidAnchor() public {
        bytes32 cidHash = keccak256(bytes(attendanceCid));

        vm.prank(coordinator);
        trialManager.logAttendance(participantA, 1, 1, 77, attendanceCid, cidHash);

        vm.prank(coordinator);
        vm.expectRevert(TrialManager.ReplayDetected.selector);
        trialManager.logAttendance(participantA, 1, 1, 77, attendanceCid, cidHash);
    }

    function test_InvalidCidHashBlocked() public {
        vm.prank(coordinator);
        vm.expectRevert(TrialManager.InvalidCidHash.selector);
        trialManager.logAttendance(participantA, 1, 2, 88, attendanceCid, bytes32(uint256(1)));
    }

    function test_CanonicalRewardFlowMintsToken() public {
        uint256 claimId = _prepareCanonicalClaim(participantA, 1, 5, 25, 12);
        vm.roll(block.number + 4);
        incentiveManager.executeReward(claimId);

        require(token.balanceOf(participantA) == 25 ether, "token balance mismatch");
    }

    function test_RewardCannotBeProposedWithoutAttendance() public {
        bytes32 decisionHash = keccak256(bytes(incentiveCid));

        vm.prank(coordinator);
        vm.expectRevert(IncentiveManager.AttendanceMissing.selector);
        incentiveManager.proposeReward(participantA, 1, 9, 10, 4, incentiveCid, decisionHash, 201);
    }

    function test_OnlyValidatorsCanApproveReward() public {
        uint256 claimId = _prepareRewardWithoutQuorum(participantA, 1, 7, 15, 8);

        vm.prank(participantA);
        vm.expectRevert(IncentiveManager.Unauthorized.selector);
        incentiveManager.approveReward(claimId);
    }

    function test_ConflictingRewardExecutionBlocked() public {
        bytes32 attendanceHash = keccak256(bytes(attendanceCid));
        vm.prank(coordinator);
        trialManager.logAttendance(participantA, 2, 11, 301, attendanceCid, attendanceHash);

        bytes32 decisionHash = keccak256(bytes(incentiveCid));
        vm.prank(coordinator);
        uint256 claimA = incentiveManager.proposeReward(participantA, 2, 11, 10, 4, incentiveCid, decisionHash, 302);
        vm.prank(coordinator);
        uint256 claimB = incentiveManager.proposeReward(participantA, 2, 11, 12, 8, incentiveCid, decisionHash, 303);

        _approveByQuorum(claimA);
        _approveByQuorum(claimB);
        vm.roll(block.number + 4);
        incentiveManager.executeReward(claimA);

        vm.expectRevert(IncentiveManager.VisitAlreadyRewarded.selector);
        incentiveManager.executeReward(claimB);
    }

    function test_CancelledClaimCannotExecute() public {
        uint256 claimId = _prepareCanonicalClaim(participantA, 3, 14, 11, 6);
        vm.prank(coordinator);
        incentiveManager.cancelClaim(claimId);
        vm.roll(block.number + 4);

        vm.expectRevert(IncentiveManager.ClaimCancelled.selector);
        incentiveManager.executeReward(claimId);
    }

    function testFuzz_VisitCanOnlyBeRewardedOnceWithTokenMint(uint96 rawA, uint96 rawB) public {
        uint128 amountA = uint128(uint256(rawA) % 500 + 1);
        uint128 amountB = uint128(uint256(rawB) % 500 + 1);

        bytes32 attendanceHash = keccak256(bytes(attendanceCid));
        vm.prank(coordinator);
        trialManager.logAttendance(participantA, 5, 2, 501, attendanceCid, attendanceHash);

        bytes32 decisionHash = keccak256(bytes(incentiveCid));
        vm.prank(coordinator);
        uint256 claimA = incentiveManager.proposeReward(participantA, 5, 2, amountA, 6, incentiveCid, decisionHash, 502);
        vm.prank(coordinator);
        uint256 claimB = incentiveManager.proposeReward(participantA, 5, 2, amountB, 12, incentiveCid, decisionHash, 503);

        _approveByQuorum(claimA);
        _approveByQuorum(claimB);
        vm.roll(block.number + 4);
        incentiveManager.executeReward(claimA);

        vm.expectRevert(IncentiveManager.VisitAlreadyRewarded.selector);
        incentiveManager.executeReward(claimB);
    }

    function _prepareCanonicalClaim(
        address participant,
        uint256 trialId,
        uint256 visitId,
        uint128 amount,
        uint128 incentiveLevel
    ) internal returns (uint256 claimId) {
        claimId = _prepareRewardWithoutQuorum(participant, trialId, visitId, amount, incentiveLevel);
        _approveByQuorum(claimId);
    }

    function _prepareRewardWithoutQuorum(
        address participant,
        uint256 trialId,
        uint256 visitId,
        uint128 amount,
        uint128 incentiveLevel
    ) internal returns (uint256 claimId) {
        bytes32 attendanceHash = keccak256(bytes(attendanceCid));
        vm.prank(coordinator);
        trialManager.logAttendance(participant, trialId, visitId, 101, attendanceCid, attendanceHash);

        bytes32 modelHash = keccak256("model-round");
        vm.prank(coordinator);
        trialManager.anchorModelUpdate(modelCid, keccak256(bytes(modelCid)), modelHash, 1, 102);

        vm.prank(coordinator);
        trialManager.logIncentiveDecision(1, incentiveLevel, incentiveCid, keccak256(bytes(incentiveCid)), 103);

        vm.prank(coordinator);
        claimId = incentiveManager.proposeReward(
            participant,
            trialId,
            visitId,
            amount,
            incentiveLevel,
            incentiveCid,
            keccak256(bytes(incentiveCid)),
            104
        );
    }

    function _approveByQuorum(uint256 claimId) internal {
        vm.prank(validatorA);
        incentiveManager.approveReward(claimId);
        vm.prank(validatorB);
        incentiveManager.approveReward(claimId);
    }

    function _readArtifact(uint256 artifactId)
        internal
        view
        returns (
            string memory artifactType,
            string memory cid,
            bytes32 cidHash,
            bytes32 metadataHash,
            uint64 logicalRef,
            uint64 anchoredBlock,
            address actor
        )
    {
        (artifactType, cid, cidHash, metadataHash, logicalRef, anchoredBlock, actor) = trialManager.artifacts(artifactId);
    }
}
