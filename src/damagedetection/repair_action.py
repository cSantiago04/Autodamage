"""
repair_action.py
Combines the damage detector and the part segmenter into what the cost stage needs:
  1. assign_part(): which car part each damage is on (overlap of the two masks)
  2. choose_action(): the likely repair, from damage type + part + extent
     (fraction of the part the damage covers)
  3. assess(): runs both over all detections in an image

Actions use the same names as the cost estimator (src/costestimation/estimator.py):
    "refinish only", "paintless dent repair", "repair and refinish", "replace"
(plus "repair" for a flat tire: plug/patch).

No dataset labels repair actions, so the rules and thresholds below are starting
guesses. They can be sanity-checked against the hand severity labels with
src/damagedetection/evaluate_repair_action.py, and ultimately against real quotes.
"""

import numpy as np

REFINISH, PDR, REPAIR, REPLACE, TIRE_REPAIR = (
    "refinish only", "paintless dent repair", "repair and refinish", "replace", "repair")
# Least to most work; used to compare an action with a minor/moderate/severe label
ACTION_LEVEL = {REFINISH: 0, PDR: 0, TIRE_REPAIR: 1, REPAIR: 1, REPLACE: 2}

# Damage types whose repair doesn't depend on size
FIXED_ACTION = {"glass shatter": REPLACE, "lamp broken": REPLACE, "tire flat": TIRE_REPAIR}

# Unpainted parts (glass, lights) can't be dent-pulled or repainted: a crack or dent means replace
UNPAINTED_PARTS = {"front_glass", "back_glass", "front_light", "back_light"}

# Extent = fraction of the PART covered by the damage. Each list is (upper bound, action),
# checked in order; the last entry has no upper bound.
EXTENT_RULES = {
    "dent":    [(0.05, PDR), (0.20, REPAIR), (None, REPLACE)],
    "scratch": [(0.25, REFINISH), (None, REPAIR)],   # scratches rarely justify replacing a panel
    "crack":   [(0.10, REPAIR), (None, REPLACE)],    # a crack always needs at least a repair
}
# Fallback when no part was matched: fraction of the whole IMAGE (coarser: depends on
# how close the photo was taken)
IMAGE_EXTENT_RULES = {
    "dent":    [(0.03, PDR), (0.12, REPAIR), (None, REPLACE)],
    "scratch": [(0.15, REFINISH), (None, REPAIR)],
    "crack":   [(0.05, REPAIR), (None, REPLACE)],
}


def assign_part(damage_mask, parts, min_overlap=0.2):
    """
    Finds the part a damage lies on.
    Args:
        damage_mask: bool array (H, W).
        parts: list of {"part": name, "mask": bool array (H, W), ...} from predict_parts().
        min_overlap: minimum fraction of the damage that must fall on a part to count.
    Returns dict:
        part: best-matching part name, or None
        overlap: fraction of the damage on that part
        extent: fraction of that part covered by the damage
        also_on: other parts that hold >= min_overlap of the damage (damage spanning panels)
    """
    damage_area = int(damage_mask.sum())
    if damage_area == 0 or not parts:
        return {"part": None, "overlap": 0.0, "extent": None, "also_on": []}

    scored = []
    for p in parts:
        inter = int(np.logical_and(damage_mask, p["mask"]).sum())
        if inter:
            scored.append((inter / damage_area, inter / max(int(p["mask"].sum()), 1), p["part"]))
    scored.sort(reverse=True)

    hits = [s for s in scored if s[0] >= min_overlap]
    if not hits:
        return {"part": None, "overlap": round(scored[0][0], 3) if scored else 0.0,
                "extent": None, "also_on": []}
    overlap, extent, name = hits[0]
    return {
        "part": name,
        "overlap": round(overlap, 3),
        "extent": round(min(extent, 1.0), 4),
        "also_on": sorted({h[2] for h in hits[1:] if h[2] != name}),
    }


def _apply(rules, value):
    for upper, action in rules:
        if upper is None or value < upper:
            return action


def choose_action(damage_type, part=None, extent=None, image_fraction=None,
                  extent_rules=None, image_extent_rules=None):
    """
    Returns (action, basis). basis says what the decision was based on:
    "damage type", "unpainted part", "part extent", "image extent", or "default".
    """
    extent_rules = extent_rules or EXTENT_RULES
    image_extent_rules = image_extent_rules or IMAGE_EXTENT_RULES

    if damage_type in FIXED_ACTION:
        return FIXED_ACTION[damage_type], "damage type"
    if part in UNPAINTED_PARTS and damage_type in ("crack", "dent"):
        return REPLACE, "unpainted part"
    if extent is not None and damage_type in extent_rules:
        return _apply(extent_rules[damage_type], extent), "part extent"
    if image_fraction is not None and damage_type in image_extent_rules:
        return _apply(image_extent_rules[damage_type], image_fraction), "image extent"
    return REPAIR, "default"


def assess(damages, parts, **rule_kwargs):
    """
    Args:
        damages: predict_damage(..., return_masks=True) output (each needs "mask").
        parts: predict_parts() output.
    Returns one record per damage: damage_type, confidence, bbox, part, overlap, extent,
    also_on, image_fraction, repair_action, action_basis. Masks are dropped so the
    result is JSON-friendly.
    """
    results = []
    for d in damages:
        mask = d["mask"]
        match = assign_part(mask, parts)
        image_fraction = float(mask.sum()) / mask.size
        action, basis = choose_action(d["damage_type"], match["part"], match["extent"], image_fraction,
                                      **rule_kwargs)
        results.append({
            "damage_type": d["damage_type"],
            "confidence": d.get("confidence"),
            "bbox": d.get("bbox"),
            **match,
            "image_fraction": round(image_fraction, 4),
            "repair_action": action,
            "action_basis": basis,
        })
    return results
