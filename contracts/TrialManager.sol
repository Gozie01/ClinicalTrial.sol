// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract TrialManager {
    error Unauthorized();
    error ZeroAddress();
    error UnknownParticipant();
    error ReplayDetected();
    error AttendanceAlreadyLogged();
    error ParticipantForgery();
    error InvalidCidHash();

    struct ArtifactRecord {
        string artifactType;
        string cid;
        bytes32 cidHash;
        bytes32 metadataHash;
        uint64 logicalRef;
        uint64 anchoredBlock;
        address actor;
    }

    address public immutable owner;
    uint256 public artifactCount;
    uint256 public incentiveDecisionCount;

    mapping(address => bool) public isCoordinator;
    mapping(address => bool) public isParticipant;
    mapping(bytes32 => bool) public attendanceLogged;
    mapping(bytes32 => bool) public actionDigestUsed;
    mapping(uint256 => ArtifactRecord) public artifacts;

    event CoordinatorRegistered(address indexed coordinator, bool enabled);
    event ParticipantRegistered(address indexed participant, bool enabled);
    event AttendanceLogged(
        address indexed actor,
        address indexed participant,
        uint256 indexed trialId,
        uint256 visitId,
        bytes32 visitKey,
        string cid,
        bytes32 cidHash
    );
    event ArtifactAnchored(
        uint256 indexed artifactId,
        string artifactType,
        string cid,
        bytes32 cidHash,
        bytes32 metadataHash,
        uint256 logicalRef,
        address indexed actor
    );
    event IncentiveDecisionLogged(
        uint256 indexed decisionId,
        uint256 indexed logicalStep,
        uint128 incentiveLevel,
        string cid,
        bytes32 cidHash,
        address indexed actor
    );

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier onlyOwnerOrCoordinator() {
        if (msg.sender != owner && !isCoordinator[msg.sender]) revert Unauthorized();
        _;
    }

    constructor(address[] memory initialCoordinators) {
        owner = msg.sender;
        _setCoordinators(initialCoordinators, true);
    }

    function registerCoordinator(address coordinator, bool enabled) external onlyOwner {
        _setCoordinator(coordinator, enabled);
    }

    function registerCoordinators(address[] calldata coordinators, bool enabled) external onlyOwner {
        _setCoordinators(coordinators, enabled);
    }

    function registerParticipant(address participant, bool enabled) external onlyOwner {
        if (participant == address(0)) revert ZeroAddress();
        isParticipant[participant] = enabled;
        emit ParticipantRegistered(participant, enabled);
    }

    function registerParticipants(address[] calldata participants, bool enabled) external onlyOwner {
        for (uint256 i = 0; i < participants.length; ++i) {
            if (participants[i] == address(0)) revert ZeroAddress();
            isParticipant[participants[i]] = enabled;
            emit ParticipantRegistered(participants[i], enabled);
        }
    }

    function computeVisitKey(address participant, uint256 trialId, uint256 visitId) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(participant, trialId, visitId));
    }

    function isAttendanceRecorded(bytes32 visitKey) external view returns (bool) {
        return attendanceLogged[visitKey];
    }

    function cidMatches(string memory cid, bytes32 cidHash) public pure returns (bool) {
        return keccak256(bytes(cid)) == cidHash;
    }

    function logAttendance(
        address participant,
        uint256 trialId,
        uint256 visitId,
        uint256 replayNonce,
        string calldata cid,
        bytes32 cidHash
    ) external {
        if (!isParticipant[participant]) revert UnknownParticipant();
        if (!cidMatches(cid, cidHash)) revert InvalidCidHash();
        if (msg.sender != owner && !isCoordinator[msg.sender] && msg.sender != participant) revert Unauthorized();
        if (msg.sender == participant && participant != msg.sender) revert ParticipantForgery();
        if (msg.sender != participant && msg.sender != owner && !isCoordinator[msg.sender]) revert ParticipantForgery();

        bytes32 actionDigest = keccak256(abi.encodePacked("ATTENDANCE", participant, trialId, visitId, replayNonce));
        if (actionDigestUsed[actionDigest]) revert ReplayDetected();
        bytes32 visitKey = computeVisitKey(participant, trialId, visitId);
        if (attendanceLogged[visitKey]) revert AttendanceAlreadyLogged();

        actionDigestUsed[actionDigest] = true;
        attendanceLogged[visitKey] = true;
        emit AttendanceLogged(msg.sender, participant, trialId, visitId, visitKey, cid, cidHash);
    }

    function anchorModelUpdate(
        string calldata cid,
        bytes32 cidHash,
        bytes32 modelHash,
        uint256 roundId,
        uint256 replayNonce
    ) external onlyOwnerOrCoordinator returns (uint256 artifactId) {
        return _anchorArtifact("MODEL_UPDATE", cid, cidHash, modelHash, roundId, replayNonce);
    }

    function logIncentiveDecision(
        uint256 logicalStep,
        uint128 incentiveLevel,
        string calldata cid,
        bytes32 cidHash,
        uint256 replayNonce
    ) external onlyOwnerOrCoordinator returns (uint256 decisionId) {
        bytes32 metadataHash = keccak256(abi.encodePacked("INCENTIVE", logicalStep, incentiveLevel));
        decisionId = _anchorArtifact("INCENTIVE_DECISION", cid, cidHash, metadataHash, logicalStep, replayNonce);
        emit IncentiveDecisionLogged(decisionId, logicalStep, incentiveLevel, cid, cidHash, msg.sender);
    }

    function _anchorArtifact(
        string memory artifactType,
        string calldata cid,
        bytes32 cidHash,
        bytes32 metadataHash,
        uint256 logicalRef,
        uint256 replayNonce
    ) internal returns (uint256 artifactId) {
        if (!cidMatches(cid, cidHash)) revert InvalidCidHash();
        bytes32 actionDigest = keccak256(abi.encodePacked(artifactType, cidHash, logicalRef, replayNonce));
        if (actionDigestUsed[actionDigest]) revert ReplayDetected();
        actionDigestUsed[actionDigest] = true;

        artifactId = ++artifactCount;
        artifacts[artifactId] = ArtifactRecord({
            artifactType: artifactType,
            cid: cid,
            cidHash: cidHash,
            metadataHash: metadataHash,
            // forge-lint: disable-next-line(unsafe-typecast)
            logicalRef: uint64(logicalRef),
            anchoredBlock: uint64(block.number),
            actor: msg.sender
        });
        emit ArtifactAnchored(artifactId, artifactType, cid, cidHash, metadataHash, logicalRef, msg.sender);
    }

    function _setCoordinators(address[] memory coordinators, bool enabled) internal {
        for (uint256 i = 0; i < coordinators.length; ++i) {
            _setCoordinator(coordinators[i], enabled);
        }
    }

    function _setCoordinator(address coordinator, bool enabled) internal {
        if (coordinator == address(0)) revert ZeroAddress();
        isCoordinator[coordinator] = enabled;
        emit CoordinatorRegistered(coordinator, enabled);
    }
}
