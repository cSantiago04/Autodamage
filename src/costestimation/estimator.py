"""
estimator.py
Rules-based repair cost estimate (Stage 3, v1).

Takes the damage list produced by src/gemma/describe_damage.py:
    [{"part": "driver front door", "damage_type": "dent", "severity": "moderate",
      "likely_repair": "repair and refinish"}, ...]
and returns a low/high cost range per damaged panel plus a total.

Structure mirrors a real body-shop estimate:
    labor hours x labor rate  +  parts  +  paint materials (per refinish hour)
with flat prices for paintless dent repair, windshield chips, tire plugs, etc.
All numbers live in cost_tables.json so they can be calibrated without code changes.

Demo:  python -m src.costestimation.estimator
"""

import json
import os
import re

TABLES_PATH = os.path.join(os.path.dirname(__file__), "cost_tables.json")
SEVERITIES = ["minor", "moderate", "severe"]

# Actions, ordered from least to most work (used when merging damage on one panel)
ACTION_RANK = {"refinish_only": 0, "pdr": 1, "flat_repair": 1, "repair": 2, "replace": 3}
ACTION_LABELS = {
    "refinish_only": "refinish only",
    "pdr": "paintless dent repair",
    "flat_repair": "repair",
    "repair": "repair and refinish",
    "replace": "replace",
}
# Gemma's likely_repair values (see describe_damage.REPAIR_ACTIONS) -> internal action
LIKELY_REPAIR_MAP = {
    "paintless dent repair": "pdr",
    "repair and refinish": "repair",
    "replace": "replace",
    "refinish only": "refinish_only",
}
LOCATION_WORDS = {
    "left": "left", "driver": "left", "lh": "left",
    "right": "right", "passenger": "right", "rh": "right",
    "front": "front", "rear": "rear", "back": "rear",
}

_tables_cache = {}


def load_tables(path=TABLES_PATH):
    if path not in _tables_cache:
        with open(path) as f:
            _tables_cache[path] = json.load(f)
    return _tables_cache[path]


# ---------------------------------------------------------------- parsing ---

def normalize_part(text, tables):
    """
    Maps free-text part names to a canonical part key plus a location tuple, e.g.
    "Driver-side front door" -> ("door", ("front", "left")). Returns (None, ()) if unknown.
    Longest alias wins, so "rear fender" -> quarter_panel rather than fender.
    """
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    aliases = sorted(((a, key) for key, cfg in tables["parts"].items() for a in cfg["aliases"]),
                     key=lambda x: -len(x[0]))
    for alias, key in aliases:
        pattern = r"\b" + re.escape(alias) + r"s?\b"
        if re.search(pattern, t):
            rest = re.sub(pattern, " ", t)  # don't read "rear" in "rear window" as a location
            location = tuple(sorted({LOCATION_WORDS[w] for w in rest.split() if w in LOCATION_WORDS}))
            return key, location
    return None, ()


def classify_vehicle(vehicle, tables):
    make = (vehicle or {}).get("make", "").lower().strip()
    if not make:
        return "standard"
    if any(make.startswith(m) for m in tables["exotic_makes"]):
        return "exotic"
    if any(make.startswith(m) for m in tables["luxury_makes"]):
        return "luxury"
    return "standard"


# ---------------------------------------------------------------- actions ---

def default_action(cfg, damage_type, severity):
    """Best-guess repair action from damage type + severity when none is given."""
    dt = (damage_type or "").lower()
    if "shatter" in dt or "broken" in dt:
        return "replace"
    if "flat" in dt or "puncture" in dt:
        return "flat_repair" if severity == "minor" else "replace"
    if "crack" in dt:
        if not cfg["painted"]:
            return "flat_repair" if severity == "minor" else "replace"
        return "replace" if severity == "severe" else "repair"
    if "dent" in dt:
        return {"minor": "pdr", "moderate": "repair", "severe": "replace"}[severity]
    if "scratch" in dt or "scuff" in dt:
        return "refinish_only" if severity == "minor" else "repair"
    return "replace" if severity == "severe" else "repair"


def resolve_action(cfg, action, severity, tables):
    """Downgrade/upgrade an action to one this part actually supports."""
    if action == "pdr" and not (cfg["painted"] and severity in tables["pdr_flat_cost"]):
        action = "repair"
    if action == "refinish_only" and not cfg["painted"]:
        action = "repair"
    if action == "repair" and severity in cfg.get("flat_repair_cost", {}):
        action = "flat_repair"
    if action == "repair" and severity not in cfg.get("repair_hours", {}):
        action = "replace"
    if action == "flat_repair" and severity not in cfg.get("flat_repair_cost", {}):
        action = "replace"
    return action


# ---------------------------------------------------------------- pricing ---

def _scale(rng, k):
    return [rng[0] * k, rng[1] * k]


def _add(*ranges):
    return [sum(r[0] for r in ranges), sum(r[1] for r in ranges)]


def _mul(a, b):
    return [a[0] * b[0], a[1] * b[1]]


def price_line(cfg, action, severity, rate, parts_mult, labor_mult, tables):
    zero = [0, 0]
    hours, parts, flat, refinish = zero, zero, zero, zero

    if action == "pdr":
        flat = _scale(tables["pdr_flat_cost"][severity], labor_mult)
    elif action == "flat_repair":
        flat = _scale(cfg["flat_repair_cost"][severity], labor_mult)
    elif action == "refinish_only":
        refinish = cfg["refinish_hours"]
    elif action == "repair":
        hours = cfg["repair_hours"][severity]
        refinish = cfg.get("refinish_hours", zero) if cfg["painted"] else zero
    elif action == "replace":
        hours = cfg["replace_labor_hours"]
        parts = _scale(cfg["replace_part_cost"], parts_mult)
        refinish = cfg.get("refinish_hours", zero) if cfg["painted"] else zero

    total_hours = _add(hours, refinish)
    labor = _mul(total_hours, rate)
    materials = _mul(refinish, tables["paint_materials_per_refinish_hour"])
    total = _add(labor, parts, materials, flat)

    r = lambda x: [round(x[0]), round(x[1])]
    return {
        "labor_hours": [round(total_hours[0], 1), round(total_hours[1], 1)],
        "labor": r(labor),
        "parts": r(parts),
        "paint_materials": r(materials),
        "flat_rate": r(flat),
        "total": r(total),
    }


# ---------------------------------------------------------------- main API ---

def estimate_repair(damages, vehicle=None, labor_rate=None, tables=None):
    """
    Args:
        damages: list of dicts with part, damage_type, severity, optional likely_repair.
        vehicle: optional dict with "make" (used for luxury/exotic pricing).
        labor_rate: optional override, a number or [low, high] dollars per hour.
    Returns:
        dict with line_items, total [low, high], vehicle_class, unrecognized, warnings,
        assumptions, disclaimer.
    """
    tables = tables or load_tables()
    vclass = classify_vehicle(vehicle, tables)
    mults = tables["vehicle_class"][vclass]

    if labor_rate is None:
        base_rate = tables["labor_rate_per_hour"]
    elif isinstance(labor_rate, (int, float)):
        base_rate = [labor_rate, labor_rate]
    else:
        base_rate = list(labor_rate)
    rate = _scale(base_rate, mults["labor_multiplier"])

    warnings, unrecognized = [], []
    groups = {}  # (part_key, location) -> merged damage info

    for d in damages:
        key, location = normalize_part(d.get("part", ""), tables)
        if key is None:
            unrecognized.append(d.get("part", ""))
            continue
        cfg = tables["parts"][key]

        severity = (d.get("severity") or "").lower()
        if severity not in SEVERITIES:
            warnings.append(f"Unknown severity '{d.get('severity')}' for {d.get('part')}; assumed moderate.")
            severity = "moderate"

        action = LIKELY_REPAIR_MAP.get((d.get("likely_repair") or "").lower())
        if action is None:
            action = default_action(cfg, d.get("damage_type"), severity)
        action = resolve_action(cfg, action, severity, tables)

        g = groups.setdefault((key, location), {"descriptions": [], "damage_types": [],
                                                "severity": severity, "action": action})
        g["descriptions"].append(d.get("part", key))
        if d.get("damage_type"):
            g["damage_types"].append(d["damage_type"])
        # Multiple damages on one panel = one repair at the worst level (refinish once)
        if SEVERITIES.index(severity) > SEVERITIES.index(g["severity"]):
            g["severity"] = severity
        if ACTION_RANK[action] > ACTION_RANK[g["action"]]:
            g["action"] = action

    line_items = []
    for (key, location), g in groups.items():
        cfg = tables["parts"][key]
        # Re-resolve: the merged severity may differ from the one the action was picked for
        action = resolve_action(cfg, g["action"], g["severity"], tables)
        costs = price_line(cfg, action, g["severity"], rate,
                           mults["parts_multiplier"], mults["labor_multiplier"], tables)
        line_items.append({
            "part": key,
            "location": list(location),
            "described_as": g["descriptions"],
            "damage_types": g["damage_types"],
            "severity": g["severity"],
            "action": ACTION_LABELS[action],
            **costs,
            "notes": [cfg["note"]] if "note" in cfg else [],
        })

    if unrecognized:
        warnings.append(f"Not priced (unrecognized parts): {', '.join(unrecognized)}")

    total = _add(*[li["total"] for li in line_items]) if line_items else [0, 0]
    return {
        "vehicle_class": vclass,
        "line_items": line_items,
        "total": total,
        "unrecognized": unrecognized,
        "warnings": warnings,
        "assumptions": {
            "labor_rate_per_hour": [round(rate[0]), round(rate[1])],
            "paint_materials_per_refinish_hour": tables["paint_materials_per_refinish_hour"],
            "parts_multiplier": mults["parts_multiplier"],
        },
        "disclaimer": "Rough estimate from uncalibrated rule tables. Actual quotes vary by shop, "
                      "region, parts availability, and hidden damage not visible in the photo.",
    }


def format_estimate(est):
    """Human-readable version of estimate_repair()'s output."""
    lines = [f"Vehicle class: {est['vehicle_class']}  |  labor ${est['assumptions']['labor_rate_per_hour'][0]}"
             f"-${est['assumptions']['labor_rate_per_hour'][1]}/hr", ""]
    for li in est["line_items"]:
        where = " ".join(li["location"])
        name = f"{where} {li['part'].replace('_', ' ')}".strip()
        lines.append(f"  {name:<28} {li['severity']:<9} {li['action']:<22} "
                     f"${li['total'][0]:>6,} - ${li['total'][1]:>6,}")
        for n in li["notes"]:
            lines.append(f"      note: {n}")
    lines.append(f"\n  {'ESTIMATED TOTAL':<61} ${est['total'][0]:>6,} - ${est['total'][1]:>6,}")
    for w in est["warnings"]:
        lines.append(f"  warning: {w}")
    lines.append(f"\n  {est['disclaimer']}")
    return "\n".join(lines)


if __name__ == "__main__":
    example = [
        {"part": "front bumper cover", "damage_type": "dent", "severity": "moderate",
         "likely_repair": "repair and refinish"},
        {"part": "front bumper", "damage_type": "scratch", "severity": "minor", "likely_repair": "unknown"},
        {"part": "left headlight assembly", "damage_type": "lamp broken", "severity": "severe",
         "likely_repair": "replace"},
        {"part": "driver front door", "damage_type": "dent", "severity": "minor",
         "likely_repair": "paintless dent repair"},
    ]
    for make in ("Honda", "Porsche"):
        print(f"\n===== {make} =====")
        print(format_estimate(estimate_repair(example, vehicle={"make": make})))
