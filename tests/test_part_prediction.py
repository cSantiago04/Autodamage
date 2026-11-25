import numpy as np

from src.damagedetection.part_prediction import suppress_duplicates


def part(name, conf, x1, x2):
    m = np.zeros((10, 20), dtype=bool)
    m[:, x1:x2] = True
    return {"part": name, "confidence": conf, "mask": m}


def test_same_panel_with_two_names_keeps_the_more_confident():
    out = suppress_duplicates([part("front_bumper", 0.7, 0, 10), part("back_bumper", 0.9, 0, 10)])
    assert [p["part"] for p in out] == ["back_bumper"]


def test_neighbouring_parts_are_kept():
    out = suppress_duplicates([part("front_door", 0.9, 0, 10), part("back_door", 0.8, 8, 18)])
    assert sorted(p["part"] for p in out) == ["back_door", "front_door"]
