import numpy as np

from src.damagedetection.severity import assess, assign_part, rate_severity


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
    assert m["part_fraction"] == 0.02
    assert m["also_on"] == []


def test_damage_spanning_two_parts():
    m = assign_part(box_mask(40, 10, 70, 20), [DOOR, BUMPER])  # 1/3 door, 2/3 bumper
    assert m["part"] == "front_bumper"
    assert m["also_on"] == ["front_door"]


def test_no_part_match():
    m = assign_part(box_mask(10, 10, 20, 20), [BUMPER])
    assert m["part"] is None and m["part_fraction"] is None
    assert assign_part(box_mask(10, 10, 20, 20), [])["part"] is None


def test_fixed_severity_types():
    assert rate_severity("glass shatter", part_fraction=0.01) == ("severe", "damage type")
    assert rate_severity("lamp broken") == ("severe", "damage type")
    assert rate_severity("tire flat") == ("moderate", "damage type")


def test_dent_severity_scales_with_part_fraction():
    assert rate_severity("dent", part_fraction=0.02)[0] == "minor"
    assert rate_severity("dent", part_fraction=0.10)[0] == "moderate"
    assert rate_severity("dent", part_fraction=0.50)[0] == "severe"


def test_falls_back_to_image_fraction_then_default():
    assert rate_severity("dent", image_fraction=0.20) == ("severe", "image fraction")
    assert rate_severity("mystery", image_fraction=0.5) == ("moderate", "default")


def test_custom_thresholds():
    assert rate_severity("dent", part_fraction=0.10, part_thresholds={"dent": [0.5, 0.9]})[0] == "minor"


def test_assess_end_to_end():
    damages = [{"damage_type": "dent", "confidence": 0.9, "bbox": None, "mask": box_mask(0, 0, 50, 30)},
               {"damage_type": "lamp broken", "confidence": 0.8, "bbox": None, "mask": box_mask(60, 60, 70, 70)}]
    out = assess(damages, [DOOR, BUMPER])
    assert [o["part"] for o in out] == ["front_door", "front_bumper"]
    assert out[0]["severity"] == "severe"  # 1500/5000 = 30% of the door
    assert out[1]["severity"] == "severe"
    assert "mask" not in out[0]
