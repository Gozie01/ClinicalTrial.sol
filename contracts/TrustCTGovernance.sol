// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

contract TrustCTGovernance {
    error Unauthorized();
    error ZeroAddress();
    error InvalidQuorum();
    error InvalidAmount();
    error UnknownParticipant();
    error UnknownClaim();
    error ReplayDetected();
    error AttendanceAlreadyLogged();
    error AttendanceMissing();
    error AlreadyApproved();
    error ClaimAlreadyFinalized();
    error ClaimCancelled();
    error FinalityPending();
    error QuorumNotReached();
    error VisitAlreadyRewarded();
    error ParticipantForgery();

    event ParticipantRegistered(address indexed participant, bool enabled);
    event ValidatorRegistered(address indexed validator, bool enabled);
    event AttendanceLogged(address indexed actor, address indexed participant, uint256 indexed trialId, uint256 visitId, bytes32 visitKey);
    event RewardProposed(
        uint256 indexed claimId,
        address indexed proposer,
        address indexed participant,
        uint256 trialId,
        uint256 visitId,
        uint256 amount,
        uint256 executeAfterBlock
    );
    event RewardApproved(uint256 indexed claimId, address indexed validator, uint256 approvals);
    event RewardExecuted(uint256 indexed claimId, address indexed participant, uint256 amount, bytes32 visitKey);
    event ClaimCancelledEvent(uint256 indexed claimId, address indexed actor);
    event TrialManagerUpdated(address indexed trialManager);
    event QuorumUpdated(uint32 quorum);
    event FinalityDelayUpdated(uint64 finalityDelayBlocks);

    struct Claim {
        address participant;
        address proposer;
        bytes32 visitKey;
        uint128 amount;
        uint64 proposeBlock;
        uint64 executeAfterBlock;
        uint32 approvals;
        bool executed;
        bool cancelled;
    }

    address public immutable owner;
    address public trialManager;
    uint32 public quorum;
    uint64 public finalityDelayBlocks;
    uint256 public claimCount;
    uint256 public validatorCount;

    mapping(address => bool) public isValidator;
    mapping(address => bool) public isParticipant;
    mapping(bytes32 => bool) public attendanceLogged;
    mapping(bytes32 => bool) public rewardExecutedForVisit;
    mapping(bytes32 => bool) public actionDigestUsed;
    mapping(uint256 => Claim) public claims;
    mapping(uint256 => mapping(address => bool)) public hasApproved;
    mapping(address => uint256) public rewardBalances;

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier onlyManagerOrOwner() {
        if (msg.sender != owner && msg.sender != trialManager) revert Unauthorized();
        _;
    }

    modifier onlyValidator() {
        if (!isValidator[msg.sender]) revert Unauthorized();
        _;
    }

    constructor(address initialTrialManager, address[] memory validators, uint32 initialQuorum, uint64 initialFinalityDelayBlocks) {
        if (initialTrialManager == address(0)) revert ZeroAddress();
        owner = msg.sender;
        trialManager = initialTrialManager;
        _setValidators(validators, true);
        _setQuorum(initialQuorum);
        finalityDelayBlocks = initialFinalityDelayBlocks;
        emit TrialManagerUpdated(initialTrialManager);
        emit FinalityDelayUpdated(initialFinalityDelayBlocks);
    }

    function setTrialManager(address newTrialManager) external onlyOwner {
        if (newTrialManager == address(0)) revert ZeroAddress();
        trialManager = newTrialManager;
        emit TrialManagerUpdated(newTrialManager);
    }

    function setQuorum(uint32 newQuorum) external onlyOwner {
        _setQuorum(newQuorum);
    }

    function setFinalityDelay(uint64 newFinalityDelayBlocks) external onlyOwner {
        finalityDelayBlocks = newFinalityDelayBlocks;
        emit FinalityDelayUpdated(newFinalityDelayBlocks);
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

    function registerValidator(address validator, bool enabled) external onlyOwner {
        _setValidator(validator, enabled);
    }

    function registerValidators(address[] calldata validators, bool enabled) external onlyOwner {
        _setValidators(validators, enabled);
    }

    function computeVisitKey(address participant, uint256 trialId, uint256 visitId) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(participant, trialId, visitId));
    }

    function logAttendance(address participant, uint256 trialId, uint256 visitId, uint256 replayNonce) external {
        if (!isParticipant[participant]) revert UnknownParticipant();
        if (msg.sender != owner && msg.sender != trialManager && msg.sender != participant) revert Unauthorized();
        if (msg.sender == participant && participant != msg.sender) revert ParticipantForgery();
        if (msg.sender != participant && msg.sender != owner && msg.sender != trialManager) revert ParticipantForgery();

        bytes32 actionDigest = keccak256(abi.encodePacked("ATTENDANCE", participant, trialId, visitId, replayNonce));
        if (actionDigestUsed[actionDigest]) revert ReplayDetected();
        bytes32 visitKey = computeVisitKey(participant, trialId, visitId);
        if (attendanceLogged[visitKey]) revert AttendanceAlreadyLogged();

        actionDigestUsed[actionDigest] = true;
        attendanceLogged[visitKey] = true;
        emit AttendanceLogged(msg.sender, participant, trialId, visitId, visitKey);
    }

    function proposeReward(
        address participant,
        uint256 trialId,
        uint256 visitId,
        uint128 amount,
        uint256 replayNonce
    ) external onlyManagerOrOwner returns (uint256 claimId) {
        if (!isParticipant[participant]) revert UnknownParticipant();
        if (amount == 0) revert InvalidAmount();
        bytes32 visitKey = computeVisitKey(participant, trialId, visitId);
        if (!attendanceLogged[visitKey]) revert AttendanceMissing();

        bytes32 actionDigest = keccak256(abi.encodePacked("PROPOSE", participant, trialId, visitId, amount, replayNonce));
        if (actionDigestUsed[actionDigest]) revert ReplayDetected();
        actionDigestUsed[actionDigest] = true;

        claimId = ++claimCount;
        claims[claimId] = Claim({
            participant: participant,
            proposer: msg.sender,
            visitKey: visitKey,
            amount: amount,
            proposeBlock: uint64(block.number),
            executeAfterBlock: uint64(block.number + finalityDelayBlocks),
            approvals: 0,
            executed: false,
            cancelled: false
        });

        emit RewardProposed(
            claimId,
            msg.sender,
            participant,
            trialId,
            visitId,
            amount,
            uint256(block.number + finalityDelayBlocks)
        );
    }

    function approveReward(uint256 claimId) external onlyValidator {
        Claim storage claim = claims[claimId];
        if (claim.participant == address(0)) revert UnknownClaim();
        if (claim.executed) revert ClaimAlreadyFinalized();
        if (claim.cancelled) revert ClaimCancelled();
        if (hasApproved[claimId][msg.sender]) revert AlreadyApproved();

        hasApproved[claimId][msg.sender] = true;
        claim.approvals += 1;
        emit RewardApproved(claimId, msg.sender, claim.approvals);
    }

    function cancelClaim(uint256 claimId) external onlyManagerOrOwner {
        Claim storage claim = claims[claimId];
        if (claim.participant == address(0)) revert UnknownClaim();
        if (claim.executed) revert ClaimAlreadyFinalized();
        claim.cancelled = true;
        emit ClaimCancelledEvent(claimId, msg.sender);
    }

    function executeReward(uint256 claimId) external {
        Claim storage claim = claims[claimId];
        if (claim.participant == address(0)) revert UnknownClaim();
        if (claim.executed) revert ClaimAlreadyFinalized();
        if (claim.cancelled) revert ClaimCancelled();
        if (block.number < claim.executeAfterBlock) revert FinalityPending();
        if (claim.approvals < quorum) revert QuorumNotReached();
        if (rewardExecutedForVisit[claim.visitKey]) revert VisitAlreadyRewarded();

        claim.executed = true;
        rewardExecutedForVisit[claim.visitKey] = true;
        rewardBalances[claim.participant] += claim.amount;
        emit RewardExecuted(claimId, claim.participant, claim.amount, claim.visitKey);
    }

    function getClaim(uint256 claimId) external view returns (Claim memory) {
        return claims[claimId];
    }

    function _setQuorum(uint32 newQuorum) internal {
        if (newQuorum == 0 || newQuorum > validatorCount) revert InvalidQuorum();
        quorum = newQuorum;
        emit QuorumUpdated(newQuorum);
    }

    function _setValidators(address[] memory validators, bool enabled) internal {
        for (uint256 i = 0; i < validators.length; ++i) {
            _setValidator(validators[i], enabled);
        }
    }

    function _setValidator(address validator, bool enabled) internal {
        if (validator == address(0)) revert ZeroAddress();
        bool current = isValidator[validator];
        if (current == enabled) return;
        isValidator[validator] = enabled;
        if (enabled) {
            validatorCount += 1;
        } else {
            validatorCount -= 1;
            if (quorum > validatorCount) {
                // forge-lint: disable-next-line(unsafe-typecast)
                quorum = uint32(validatorCount);
                emit QuorumUpdated(quorum);
            }
        }
        emit ValidatorRegistered(validator, enabled);
    }
}
