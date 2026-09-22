"""引擎注册表：名字能查到实现，未知名字要报错。"""

import pytest

from app.engines.advice import ADVICE_ENGINES, get_advice_engine
from app.engines.behavior import BEHAVIOR_ENGINES, get_behavior_engine
from app.engines.pose import POSE_ENGINES, get_pose_engine


@pytest.mark.parametrize("key", sorted(POSE_ENGINES))
def test_pose_engines_are_constructible(key):
    engine = get_pose_engine(key)
    assert engine.key == key
    assert isinstance(engine.available, bool)


@pytest.mark.parametrize("key", sorted(BEHAVIOR_ENGINES))
def test_behavior_engines_are_constructible(key):
    engine = get_behavior_engine(key)
    assert engine.key == key
    assert isinstance(engine.available, bool)


@pytest.mark.parametrize("key", sorted(ADVICE_ENGINES))
def test_advice_engines_are_constructible(key):
    engine = get_advice_engine(key)
    assert engine.key == key


def test_unknown_engine_raises():
    with pytest.raises(KeyError):
        get_pose_engine("does-not-exist")
    with pytest.raises(KeyError):
        get_behavior_engine("does-not-exist")
    with pytest.raises(KeyError):
        get_advice_engine("does-not-exist")


def test_six_elevator_behaviors_are_registered():
    """最初需求里的六种行为必须都在规则配置里。"""
    from app.engines.behavior.rule_engine import load_rule_config

    rules = load_rule_config().get("rules") or {}
    for key in ("fall", "fight", "kick_door", "pry_door", "push_door", "lean_door"):
        assert key in rules
