// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface ITrialManager {
    function isParticipant(address participant) external view returns (bool);
    function computeVisitKey(address participant, uint256 trialId, uint256 visitId) external pure returns (bytes32);
    function isAttendanceRecorded(bytes32 visitKey) external view returns (bool);
}

interface ICTToken {
    function mint(address to, uint256 amount) external;
}

contract IncentiveManager {
    error Unauthorized();
    error ZeroAddress();
    error InvalidQuorum();
    error InvalidAmount();
    error UnknownParticipant();
    error UnknownClaim();
    error ReplayDetected();
    error AttendanceMissing();
    error AlreadyApproved();
    error ClaimAlreadyFinalized();
    error ClaimCancelled();
    error FinalityPending();
    error QuorumNotReached();
    error VisitAlreadyRewarded();
    error InvalidCidHash();

    struct Claim {
        address participant;
        address proposer;
        bytes32 visitKey;
        uint128 amount;
        uint128 incentiveLevel;
        uint64 proposeBlock;
        uint64 executeAfterBlock;
        uint32 approvals;
        bool executed;
        bool cancelled;
        bytes32 decisionCidHash;
    }

    address public immutable owner;
    address public incentiveOperator;
    uint32 public quorum;
    uint64 public finalityDelayBlocks;
    uint256 public claimCount;
    uint256 public validatorCount;

    ITrialManager public immutable trialManager;
    ICTToken public immutable ctToken;

    mapping(address => bool) public isValidator;
    mapping(bytes32 => bool) public rewardExecutedForVisit;
    mapping(bytes32 => bool) public actionDigestUsed;
    mapping(uint256 => Claim) public claims;
    mapping(uint256 => mapping(address => bool)) public hasApproved;

    event IncentiveOperatorUpdated(address indexed operator);
    event ValidatorRegistered(address indexed validator, bool enabled);
    event QuorumUpdated(uint32 quorum);
    event FinalityDelayUpdated(uint64 finalityDelayBlocks);
    event RewardProposed(
        uint256 indexed claimId,
        address indexed proposer,
        address indexed participant,
        uint256 trialId,
        uint256 visitId,
        uint256 amount,
        uint256 incentiveLevel,
        bytes32 decisionCidHash,
        uint256 executeAfterBlock
    );
    event RewardApproved(uint256 indexed claimId, address indexed validator, uint256 approvals);
    event RewardExecuted(
        uint256 indexed claimId,
        address indexed participant,
        uint256 amount,
        uint256 incentiveLevel,
        bytes32 visitKey
    );
    event ClaimCancelledEvent(uint256 indexed claimId, address indexed actor);

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized();
        _;
    }

    modifier onlyOperatorOrOwner() {
        if (msg.sender != owner && msg.sender != incentiveOperator) revert Unauthorized();
        _;
    }

    modifier onlyValidator() {
        if (!isValidator[msg.sender]) revert Unauthorized();
        _;
    }

    constructor(
        address trialManagerAddress,
        address tokenAddress,
        address initialOperator,
        address[] memory validators,
        uint32 initialQuorum,
        uint64 initialFinalityDelayBlocks
    ) {
        if (trialManagerAddress == address(0) || tokenAddress == address(0) || initialOperator == address(0)) {
            revert ZeroAddress();
        }
        owner = msg.sender;
        trialManager = ITrialManager(trialManagerAddress);
        ctToken = ICTToken(tokenAddress);
        incentiveOperator = initialOperator;
        emit IncentiveOperatorUpdated(initialOperator);
        _setValidators(validators, true);
        _setQuorum(initialQuorum);
        finalityDelayBlocks = initialFinalityDelayBlocks;
        emit FinalityDelayUpdated(initialFinalityDelayBlocks);
    }

    function setIncentiveOperator(address newOperator) external onlyOwner {
        if (newOperator == address(0)) revert ZeroAddress();
        incentiveOperator = newOperator;
        emit IncentiveOperatorUpdated(newOperator);
    }

    function setQuorum(uint32 newQuorum) external onlyOwner {
        _setQuorum(newQuorum);
    }

    function setFinalityDelay(uint64 newFinalityDelayBlocks) external onlyOwner {
        finalityDelayBlocks = newFinalityDelayBlocks;
        emit FinalityDelayUpdated(newFinalityDelayBlocks);
    }

    function registerValidator(address validator, bool enabled) external onlyOwner {
        _setValidator(validator, enabled);
    }

    function registerValidators(address[] calldata validators, bool enabled) external onlyOwner {
        _setValidators(validators, enabled);
    }

    function cidMatches(string memory cid, bytes32 cidHash) public pure returns (bool) {
        return keccak256(bytes(cid)) == cidHash;
    }

    function proposeReward(
        address participant,
        uint256 trialId,
        uint256 visitId,
        uint128 amount,
        uint128 incentiveLevel,
        string calldata decisionCid,
        bytes32 decisionCidHash,
        uint256 replayNonce
    ) external onlyOperatorOrOwner returns (uint256 claimId) {
        if (!trialManager.isParticipant(participant)) revert UnknownParticipant();
        if (amount == 0) revert InvalidAmount();
        if (!cidMatches(decisionCid, decisionCidHash)) revert InvalidCidHash();

        bytes32 visitKey = trialManager.computeVisitKey(participant, trialId, visitId);
        if (!trialManager.isAttendanceRecorded(visitKey)) revert AttendanceMissing();

        bytes32 actionDigest = keccak256(
            abi.encodePacked("PROPOSE", participant, trialId, visitId, amount, incentiveLevel, decisionCidHash, replayNonce)
        );
        if (actionDigestUsed[actionDigest]) revert ReplayDetected();
        actionDigestUsed[actionDigest] = true;

        claimId = ++claimCount;
        claims[claimId] = Claim({
            participant: participant,
            proposer: msg.sender,
            visitKey: visitKey,
            amount: amount,
            incentiveLevel: incentiveLevel,
            proposeBlock: uint64(block.number),
            executeAfterBlock: uint64(block.number + finalityDelayBlocks),
            approvals: 0,
            executed: false,
            cancelled: false,
            decisionCidHash: decisionCidHash
        });

        emit RewardProposed(
            claimId,
            msg.sender,
            participant,
            trialId,
            visitId,
            amount,
            incentiveLevel,
            decisionCidHash,
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

    function cancelClaim(uint256 claimId) external onlyOperatorOrOwner {
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
        ctToken.mint(claim.participant, uint256(claim.amount) * 1e18);
        emit RewardExecuted(claimId, claim.participant, claim.amount, claim.incentiveLevel, claim.visitKey);
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
