import numpy as np

from src.costestimation.estimator import LIKELY_REPAIR_MAP
from src.damagedetection.repair_action import (
    EXTENT_RULES, FIXED_ACTION, IMAGE_EXTENT_RULES, PDR, REFINISH, REPAIR, REPLACE, TIRE_REPAIR,
    assess, assign_part, choose_action,
)


def box_mask(x1, y1, x2, y2, shape=(100, 100)):
    m = np.zeros(shape, dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


DOOR = {"part": "front_door", "mask": box_mask(0, 0, 50, 100)}      # 5000 px
BUMPER = {"part": "front_bumper", "mask": box_mask(50, 0, 100, 100)}  # 5000 px


def test_damage_assigned_to_part_it_sits_on():
    m = assign_part(box_mask(10, 10, 20, 20), [DOOR, BUMPER])  # 100 px, all on the door
    assert m["part"] == "front_door"
    assert m["overlap"] == 1.0
    assert m["extent"] == 0.02
    assert m["also_on"] == []


def test_damage_spanning_two_parts():
    m = assign_part(box_mask(40, 10, 70, 20), [DOOR, BUMPER])  # 1/3 door, 2/3 bumper
    assert m["part"] == "front_bumper"
    assert m["also_on"] == ["front_door"]


def test_no_part_match():
    assert assign_part(box_mask(10, 10, 20, 20), [BUMPER])["part"] is None
    assert assign_part(box_mask(10, 10, 20, 20), [])["extent"] is None


def test_fixed_actions():
    assert choose_action("glass shatter", extent=0.01) == (REPLACE, "damage type")
    assert choose_action("lamp broken") == (REPLACE, "damage type")
    assert choose_action("tire flat") == (TIRE_REPAIR, "damage type")


def test_dent_action_scales_with_extent():
    assert choose_action("dent", "front_door", extent=0.02)[0] == PDR
    assert choose_action("dent", "front_door", extent=0.10)[0] == REPAIR
    assert choose_action("dent", "front_door", extent=0.50)[0] == REPLACE


def test_scratch_never_replaces_and_crack_is_never_cosmetic():
    assert choose_action("scratch", "hood", extent=0.05)[0] == REFINISH
    assert choose_action("scratch", "hood", extent=0.95)[0] == REPAIR
    assert choose_action("crack", "front_bumper", extent=0.01)[0] == REPAIR


def test_unpainted_parts_get_replaced():
    assert choose_action("crack", "front_light", extent=0.01) == (REPLACE, "unpainted part")
    assert choose_action("dent", "back_glass", extent=0.01)[0] == REPLACE


def test_falls_back_to_image_extent_then_default():
    assert choose_action("dent", image_fraction=0.20) == (REPLACE, "image extent")
    assert choose_action("mystery", image_fraction=0.5) == (REPAIR, "default")


def test_custom_rules():
    assert choose_action("dent", "hood", extent=0.10, extent_rules={"dent": [(0.5, PDR), (None, REPLACE)]})[0] == PDR


def test_actions_are_ones_the_cost_estimator_understands():
    actions = {a for rules in (EXTENT_RULES, IMAGE_EXTENT_RULES) for r in rules.values() for _, a in r}
    actions |= set(FIXED_ACTION.values()) - {TIRE_REPAIR}  # tire "repair" -> estimator's flat repair via defaults
    assert actions <= set(LIKELY_REPAIR_MAP)


def test_assess_end_to_end():
    damages = [{"damage_type": "dent", "confidence": 0.9, "bbox": None, "mask": box_mask(0, 0, 50, 30)},
               {"damage_type": "lamp broken", "confidence": 0.8, "bbox": None, "mask": box_mask(60, 60, 70, 70)}]
    out = assess(damages, [DOOR, BUMPER])
    assert [o["part"] for o in out] == ["front_door", "front_bumper"]
    assert out[0]["repair_action"] == REPLACE  # 1500/5000 = 30% of the door
    assert out[1]["repair_action"] == REPLACE
    assert "mask" not in out[0]


def test_collision_context_signals():
    from src.damagedetection.repair_action import collision_context
    assert collision_context(["dent", "lamp broken"])
    assert collision_context(["glass shatter"])
    assert collision_context(["dent", "scratch", "dent", "crack"])  # 4 separate damages
    assert not collision_context(["dent", "scratch"])


def test_collision_escalates_size_based_actions_one_level():
    assert choose_action("dent", "hood", extent=0.02, collision=True) == (REPAIR, "part extent + collision")
    assert choose_action("dent", "hood", extent=0.10, collision=True)[0] == REPLACE
    assert choose_action("scratch", "hood", extent=0.05, collision=True)[0] == REPAIR
    assert choose_action("crack", image_fraction=0.01, collision=True) == (REPLACE, "image extent + collision")


def test_collision_leaves_fixed_actions_alone():
    assert choose_action("tire flat", collision=True) == (TIRE_REPAIR, "damage type")
    assert choose_action("lamp broken", collision=True) == (REPLACE, "damage type")


def test_assess_applies_collision_context_to_whole_photo():
    small_dent = {"damage_type": "dent", "confidence": 0.9, "bbox": None, "mask": box_mask(0, 0, 10, 10)}
    lamp = {"damage_type": "lamp broken", "confidence": 0.9, "bbox": None, "mask": box_mask(60, 60, 70, 70)}
    assert assess([small_dent], [DOOR])[0]["repair_action"] == PDR
    assert assess([small_dent, lamp], [DOOR, BUMPER])[0]["repair_action"] == REPAIR
    assert assess([small_dent, lamp], [DOOR, BUMPER], use_collision_context=False)[0]["repair_action"] == PDR
