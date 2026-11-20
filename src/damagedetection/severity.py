"""
severity.py
Combines the damage detector and the part segmenter:
  1. assign_part(): which car part each damage is on (overlap of the two masks)
  2. rate_severity(): minor / moderate / severe from damage type + how much of the
     part is damaged
  3. assess(): runs both over all detections in an image

Severity is rules-based (v1). No dataset labels severity, so the thresholds below
are starting guesses, to be measured and tuned against hand labels with
src/damagedetection/evaluate_severity.py.
"""

import numpy as np

SEVERITIES = ["minor", "moderate", "severe"]

# Damage types whose repair doesn't depend on size: broken glass/lamps get replaced,
# and a flat tire is "moderate" (plug or replace can't be told apart from a photo).
FIXED_SEVERITY = {"glass shatter": "severe", "lamp broken": "severe", "tire flat": "moderate"}

# Fraction of the PART covered by the damage: [minor/moderate cutoff, moderate/severe cutoff]
PART_FRACTION_THRESHOLDS = {
    "dent": [0.05, 0.20],
    "scratch": [0.10, 0.40],
    "crack": [0.02, 0.10],
}
# Fallback when no part could be matched: fraction of the whole IMAGE. Coarser, because
# it depends on how close the photo was taken.
IMAGE_FRACTION_THRESHOLDS = {
    "dent": [0.03, 0.12],
    "scratch": [0.05, 0.20],
    "crack": [0.01, 0.05],
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
        part_fraction: fraction of that part covered by the damage (severity input)
        also_on: other parts that hold >= min_overlap of the damage (damage spanning panels)
    """
    damage_area = int(damage_mask.sum())
    if damage_area == 0 or not parts:
        return {"part": None, "overlap": 0.0, "part_fraction": None, "also_on": []}

    scored = []
    for p in parts:
        inter = int(np.logical_and(damage_mask, p["mask"]).sum())
        if inter:
            scored.append((inter / damage_area, inter / max(int(p["mask"].sum()), 1), p["part"]))
    scored.sort(reverse=True)

    hits = [s for s in scored if s[0] >= min_overlap]
    if not hits:
        return {"part": None, "overlap": round(scored[0][0], 3) if scored else 0.0,
                "part_fraction": None, "also_on": []}
    overlap, part_fraction, name = hits[0]
    return {
        "part": name,
        "overlap": round(overlap, 3),
        "part_fraction": round(min(part_fraction, 1.0), 4),
        "also_on": sorted({h[2] for h in hits[1:] if h[2] != name}),
    }


def _bucket(value, cutoffs):
    if value < cutoffs[0]:
        return "minor"
    if value < cutoffs[1]:
        return "moderate"
    return "severe"


def rate_severity(damage_type, part_fraction=None, image_fraction=None,
                  part_thresholds=None, image_thresholds=None):
    """
    Returns (severity, basis). basis says what the decision was based on:
    "damage type", "part fraction", "image fraction", or "default".
    """
    part_thresholds = part_thresholds or PART_FRACTION_THRESHOLDS
    image_thresholds = image_thresholds or IMAGE_FRACTION_THRESHOLDS

    if damage_type in FIXED_SEVERITY:
        return FIXED_SEVERITY[damage_type], "damage type"
    if part_fraction is not None and damage_type in part_thresholds:
        return _bucket(part_fraction, part_thresholds[damage_type]), "part fraction"
    if image_fraction is not None and damage_type in image_thresholds:
        return _bucket(image_fraction, image_thresholds[damage_type]), "image fraction"
    return "moderate", "default"


def assess(damages, parts, **severity_kwargs):
    """
    Args:
        damages: predict_damage(..., return_masks=True) output (each needs "mask").
        parts: predict_parts() output.
    Returns one record per damage: type, confidence, bbox, part, overlap, part_fraction,
    also_on, severity, severity_basis. Masks are dropped so the result is JSON-friendly.
    """
    results = []
    for d in damages:
        mask = d["mask"]
        match = assign_part(mask, parts)
        image_fraction = float(mask.sum()) / mask.size
        severity, basis = rate_severity(d["damage_type"], match["part_fraction"], image_fraction,
                                        **severity_kwargs)
        results.append({
            "damage_type": d["damage_type"],
            "confidence": d.get("confidence"),
            "bbox": d.get("bbox"),
            **match,
            "image_fraction": round(image_fraction, 4),
            "severity": severity,
            "severity_basis": basis,
        })
    return results
