from src.costestimation.estimator import (
    default_action, estimate_repair, load_tables, normalize_part, resolve_action,
)

T = load_tables()


def item(part, damage_type="dent", severity="moderate", likely_repair="unknown"):
    return {"part": part, "damage_type": damage_type, "severity": severity, "likely_repair": likely_repair}


# --- part normalization -------------------------------------------------------

def test_aliases_map_to_canonical_parts():
    assert normalize_part("Front bumper cover", T)[0] == "bumper"
    assert normalize_part("left headlight assembly", T)[0] == "headlight"
    assert normalize_part("tailgate", T)[0] == "trunk_lid"


def test_longest_alias_wins():
    assert normalize_part("rear fender", T)[0] == "quarter_panel"
    assert normalize_part("rear windshield", T)[0] == "rear_window"
    assert normalize_part("door glass", T)[0] == "side_window"


def test_location_parsing():
    assert normalize_part("driver front door", T) == ("door", ("front", "left"))
    assert normalize_part("Passenger-side rear door", T) == ("door", ("rear", "right"))
    # "rear" belongs to the part name here, not a location
    assert normalize_part("rear window", T) == ("rear_window", ())


def test_unknown_part():
    assert normalize_part("flux capacitor", T) == (None, ())


# --- actions --------------------------------------------------------------------

def test_default_actions():
    bumper, headlight, tire = T["parts"]["bumper"], T["parts"]["headlight"], T["parts"]["tire"]
    assert default_action(bumper, "dent", "minor") == "pdr"
    assert default_action(bumper, "dent", "severe") == "replace"
    assert default_action(bumper, "scratch", "minor") == "refinish_only"
    assert default_action(headlight, "lamp broken", "minor") == "replace"
    assert default_action(tire, "tire flat", "minor") == "flat_repair"


def test_resolve_falls_back_to_supported_action():
    # Grille has no repair hours -> must be replaced
    assert resolve_action(T["parts"]["grille"], "repair", "moderate", T) == "replace"
    # Windshield chip: "repair" becomes the flat chip-repair price
    assert resolve_action(T["parts"]["windshield"], "repair", "minor", T) == "flat_repair"
    # Unpainted part can't be PDR'd
    assert resolve_action(T["parts"]["taillight"], "pdr", "minor", T) == "replace"


# --- estimates ------------------------------------------------------------------

def test_damage_on_same_panel_is_merged():
    est = estimate_repair([item("front bumper", "dent", "moderate"),
                           item("front bumper cover", "scratch", "minor")])
    assert len(est["line_items"]) == 1
    assert est["line_items"][0]["severity"] == "moderate"


def test_different_sides_are_separate_lines():
    est = estimate_repair([item("driver front door"), item("passenger front door")])
    assert len(est["line_items"]) == 2


def test_vehicle_class_raises_cost():
    damages = [item("hood", "dent", "severe", "replace")]
    std = estimate_repair(damages, vehicle={"make": "Honda"})
    lux = estimate_repair(damages, vehicle={"make": "BMW"})
    exo = estimate_repair(damages, vehicle={"make": "Lamborghini"})
    assert (std["vehicle_class"], lux["vehicle_class"], exo["vehicle_class"]) == ("standard", "luxury", "exotic")
    assert std["total"][1] < lux["total"][1] < exo["total"][1]


def test_unrecognized_part_warns_instead_of_crashing():
    est = estimate_repair([item("flux capacitor"), item("hood")])
    assert est["unrecognized"] == ["flux capacitor"]
    assert len(est["line_items"]) == 1
    assert any("flux capacitor" in w for w in est["warnings"])


def test_bad_severity_defaults_to_moderate():
    est = estimate_repair([item("hood", severity="catastrophic")])
    assert est["line_items"][0]["severity"] == "moderate"
    assert est["warnings"]


def test_labor_rate_override():
    damages = [item("fender", "dent", "moderate", "repair and refinish")]
    cheap = estimate_repair(damages, labor_rate=50)
    pricey = estimate_repair(damages, labor_rate=150)
    assert cheap["total"][1] < pricey["total"][0]


def test_low_never_exceeds_high_for_every_part_action_severity():
    for part in T["parts"]:
        for repair in ("paintless dent repair", "repair and refinish", "replace", "refinish only", "unknown"):
            for sev in ("minor", "moderate", "severe"):
                est = estimate_repair([item(part.replace("_", " "), "dent", sev, repair)])
                for li in est["line_items"]:
                    for k in ("labor", "parts", "paint_materials", "flat_rate", "total"):
                        assert 0 <= li[k][0] <= li[k][1], (part, repair, sev, k)


def test_empty_input():
    assert estimate_repair([])["total"] == [0, 0]
