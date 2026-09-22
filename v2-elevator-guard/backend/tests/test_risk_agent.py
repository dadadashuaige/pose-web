"""RiskAgent 的确定性行为。"""

from app.services.risk_agent import RiskAgent


def test_fall_at_high_confidence_is_high_risk():
    result = RiskAgent().evaluate("fall", 0.95, 1, 0.9)
    assert result.level == "高风险"
    assert result.score >= 0.7


def test_unknown_behavior_does_not_create_risk():
    result = RiskAgent().evaluate("normal/unknown", 0.99, 1, 0.9)
    assert result.level == "低风险"


def test_without_behavior_conclusion_risk_stays_low():
    """行为引擎没给出结论时，不应该凭空制造风险。"""
    result = RiskAgent().evaluate("unanalyzed", 0.9, 6, 0.9)
    assert result.level == "低风险"
    assert result.score <= 0.2


def test_crowd_and_low_quality_add_to_score():
    agent = RiskAgent()
    single = agent.evaluate("lean_door", 0.8, 1, 0.9)
    crowded = agent.evaluate("lean_door", 0.8, 6, 0.2)
    assert crowded.score > single.score
