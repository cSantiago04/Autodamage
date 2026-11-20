from unittest.mock import patch

import pytest

from src.carclassification.vin_decoder import decode_vin, validate_vin

GOOD_VIN = "1HGCM82633A004352"  # widely used public sample VIN (2003 Honda Accord)


def test_valid_vin():
    assert validate_vin(GOOD_VIN) == (True, "")
    assert validate_vin(" 1hgcm82633a004352 ")[0]  # trims and uppercases


def test_wrong_length():
    ok, reason = validate_vin("1HGCM82633A00435")
    assert not ok and "17" in reason


def test_forbidden_letters():
    ok, reason = validate_vin("1HGCM82633A00435O")
    assert not ok and "O" in reason


def test_typo_fails_check_digit():
    ok, reason = validate_vin("1HGCM82633A004353")
    assert not ok and "Check digit" in reason


def test_decode_rejects_invalid_vin_without_network_call():
    with patch("src.carclassification.vin_decoder.requests.get") as get:
        with pytest.raises(ValueError):
            decode_vin("NOTAVIN")
        get.assert_not_called()


def test_decode_parses_vpic_response():
    fake = {"Results": [{"ModelYear": "2003", "Make": "HONDA", "Model": "Accord", "Trim": "EX",
                         "BodyClass": "Coupe", "VehicleType": "PASSENGER CAR", "ErrorCode": "0"}]}
    with patch("src.carclassification.vin_decoder.requests.get") as get:
        get.return_value.json.return_value = fake
        car = decode_vin(GOOD_VIN)
    assert (car["year"], car["make"], car["model"], car["trim"]) == (2003, "HONDA", "Accord", "EX")
    assert car["warnings"] is None
