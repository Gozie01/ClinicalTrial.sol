using TrustCTGovernance as governance;

methods {
    function rewardExecutedForVisit(bytes32) external returns (bool) envfree;
    function rewardBalances(address) external returns (uint256) envfree;
    function computeVisitKey(address,uint256,uint256) external returns (bytes32) envfree;
}

rule visit_is_not_rewarded_twice(address participant, uint256 trialId, uint256 visitId) {
    bytes32 visitKey = governance.computeVisitKey(participant, trialId, visitId);
    require governance.rewardExecutedForVisit(visitKey);
    assert governance.rewardBalances(participant) >= 0;
}
