"""
vin_decoder.py
Identifies the vehicle from its VIN using NHTSA's free vPIC API (no key needed).
This replaces guessing make/model/year from the photo: the user knows their VIN,
and it gives the exact year, make, model, trim, and body style.

    validate_vin("1HGCM82633A004352")  -> (True, "")
    decode_vin("1HGCM82633A004352")    -> {"year": 2003, "make": "HONDA", "model": "Accord", ...}

CLI:  python -m src.carclassification.vin_decoder 1HGCM82633A004352
"""

import sys

import requests

VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"

# ISO 3779 / 49 CFR 565 check-digit tables (VINs never contain I, O, or Q)
_TRANSLIT = {**{str(d): d for d in range(10)},
             **dict(zip("ABCDEFGH", range(1, 9))), **dict(zip("JKLMN", range(1, 6))),
             "P": 7, "R": 9, **dict(zip("STUVWXYZ", range(2, 10)))}
_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def validate_vin(vin: str):
    """Returns (is_valid, reason). Checks length, characters, and the check digit (position 9)."""
    vin = vin.strip().upper()
    if len(vin) != 17:
        return False, f"VIN must be 17 characters (got {len(vin)})."
    bad = sorted({c for c in vin if c not in _TRANSLIT})
    if bad:
        return False, f"Invalid characters: {', '.join(bad)} (VINs never use I, O, or Q)."
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS))
    expected = "X" if total % 11 == 10 else str(total % 11)
    if vin[8] != expected:
        return False, "Check digit doesn't match. There's probably a typo in the VIN."
    return True, ""


def decode_vin(vin: str, timeout=10) -> dict:
    """
    Decodes a VIN via NHTSA vPIC. Raises ValueError for an invalid VIN.
    Returns year, make, model, trim, body_style, vehicle_type, plus any NHTSA warnings.
    """
    vin = vin.strip().upper()
    ok, reason = validate_vin(vin)
    if not ok:
        raise ValueError(reason)

    resp = requests.get(VPIC_URL.format(vin=vin), timeout=timeout)
    resp.raise_for_status()
    r = resp.json()["Results"][0]

    def val(key):
        v = (r.get(key) or "").strip()
        return v or None

    year = val("ModelYear")
    return {
        "vin": vin,
        "year": int(year) if year and year.isdigit() else None,
        "make": val("Make"),
        "model": val("Model"),
        "trim": val("Trim") or val("Series"),
        "body_style": val("BodyClass"),
        "vehicle_type": val("VehicleType"),
        "doors": val("Doors"),
        "drive_type": val("DriveType"),
        # ErrorCode "0" means a clean decode; anything else is a partial/uncertain decode
        "warnings": None if val("ErrorCode") in (None, "0") else val("ErrorText"),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m src.carclassification.vin_decoder <VIN>")
        sys.exit(1)
    for k, v in decode_vin(sys.argv[1]).items():
        print(f"{k:>13}: {v}")
