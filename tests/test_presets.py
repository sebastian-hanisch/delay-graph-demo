"""Jedes Preset zeigt, was sein Name und seine Hilfe behaupten (Bänder mit dem ausgelieferten Code kalibriert, bewusst weit)."""

import pytest

import dg_constants as C
from dg_evaluation import Settings, analyse_for, verdict


def _measure(p):
    params = (p["m"], p["grid"], p["contact_p"], p["similarity"], p["delay_max"], p["jitter"], p["rate_scale"], p["noise"], p["seconds"], p["seed"])
    a = analyse_for(params, Settings(min_clique=C.SINGLE_CLIQUE[p["singles"]]))
    return {"verdict": verdict(a)[1], "recall": a.dg.recall, "precision": a.dg.precision, "f1": a.dg.f1}


def test_every_preset_has_help_and_bands():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS)
    assert len(C.PRESETS) == 6


def test_preset_settings_are_within_slider_bounds():
    for p in C.PRESETS.values():
        assert C.N_CELLS_MIN <= p["m"] <= C.N_CELLS_MAX and C.GRID_MIN <= p["grid"] <= C.GRID_MAX
        assert C.CONTACT_P_MIN <= p["contact_p"] <= C.CONTACT_P_MAX and abs(p["contact_p"] * 20 - round(p["contact_p"] * 20)) < 1e-9
        assert C.SIMILARITY_MIN <= p["similarity"] <= C.SIMILARITY_MAX and abs(p["similarity"] * 20 - round(p["similarity"] * 20)) < 1e-9
        assert C.DELAY_MAX_MIN <= p["delay_max"] <= C.DELAY_MAX_MAX and C.JITTER_MIN <= p["jitter"] <= C.JITTER_MAX and abs(p["jitter"] * 2 - round(p["jitter"] * 2)) < 1e-9
        assert C.RATE_SCALE_MIN <= p["rate_scale"] <= C.RATE_SCALE_MAX and abs(p["rate_scale"] * 4 - round(p["rate_scale"] * 4)) < 1e-9
        assert C.NOISE_MIN <= p["noise"] <= C.NOISE_MAX and C.SECONDS_MIN <= p["seconds"] <= C.SECONDS_MAX and abs(p["seconds"] * 2 - round(p["seconds"] * 2)) < 1e-9
        assert p["singles"] in C.SINGLE_MODES


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_stays_inside_its_bands(name):
    measured = _measure(C.PRESETS[name])
    for key, expected in C.PRESET_EXPECTED_BANDS[name].items():
        value = measured[key]
        if key == "verdict":
            assert value in expected, f"{key}: {value}"
        else:
            lo, hi = expected
            assert lo <= value <= hi, f"{key}: {value} nicht in [{lo}, {hi}]"
