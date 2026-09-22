// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../contracts/TrustCTGovernance.sol";

contract TrustCTEchidna {
    TrustCTGovernance internal governance;
    address internal manager = address(0x100);
    address internal participant = address(0x301);
    address internal validatorA = address(0x201);
    address internal validatorB = address(0x202);

    uint256 internal canonicalTrialId = 1;
    uint256 internal canonicalVisitId = 1;
    uint256 internal canonicalClaimId;
    bool internal initialized;

    constructor() {
        address[] memory validators = new address[](2);
        validators[0] = validatorA;
        validators[1] = validatorB;
        governance = new TrustCTGovernance(manager, validators, 2, 0);
        governance.registerParticipant(participant, true);
    }

    function echidna_visit_not_rewarded_twice() public returns (bool) {
        if (!initialized) {
            governance.logAttendance(participant, canonicalTrialId, canonicalVisitId, 1);
            canonicalClaimId = governance.proposeReward(participant, canonicalTrialId, canonicalVisitId, 10, 2);
            initialized = true;
        }
        return !governance.rewardExecutedForVisit(governance.computeVisitKey(participant, canonicalTrialId, canonicalVisitId))
            || governance.rewardBalances(participant) <= 10;
    }
}
