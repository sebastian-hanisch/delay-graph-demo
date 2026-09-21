"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem OR-Demo-Portfolio, siehe tm_presets.py in template-matching-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dg_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in options:
            raise ValueError(value)
        return value
    return cast


SETTING_SPECS = {
    "n_cells_slider": SettingSpec("m", int, C.DEFAULT_N_CELLS, C.N_CELLS_MIN, C.N_CELLS_MAX),
    "grid_slider": SettingSpec("grid", int, C.DEFAULT_GRID, C.GRID_MIN, C.GRID_MAX),
    "contact_p_slider": SettingSpec("p", float, C.DEFAULT_CONTACT_P, C.CONTACT_P_MIN, C.CONTACT_P_MAX),
    "similarity_slider": SettingSpec("sim", float, C.DEFAULT_SIMILARITY, C.SIMILARITY_MIN, C.SIMILARITY_MAX),
    "delay_max_slider": SettingSpec("dmax", int, C.DEFAULT_DELAY_MAX, C.DELAY_MAX_MIN, C.DELAY_MAX_MAX),
    "jitter_slider": SettingSpec("jitter", float, C.DEFAULT_JITTER, C.JITTER_MIN, C.JITTER_MAX),
    "rate_slider": SettingSpec("rate", float, C.DEFAULT_RATE_SCALE, C.RATE_SCALE_MIN, C.RATE_SCALE_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "seconds_slider": SettingSpec("sec", float, C.DEFAULT_SECONDS, C.SECONDS_MIN, C.SECONDS_MAX),
    "singles_select": SettingSpec("singles", _choice(C.SINGLE_MODES), C.DEFAULT_SINGLES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"m": "n_cells_slider", "grid": "grid_slider", "contact_p": "contact_p_slider", "similarity": "similarity_slider", "delay_max": "delay_max_slider", "jitter": "jitter_slider",
               "rate_scale": "rate_slider", "noise": "noise_slider", "seconds": "seconds_slider", "singles": "singles_select", "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("rate_slider", 4), ("similarity_slider", 20), ("contact_p_slider", 20), ("jitter_slider", 2), ("seconds_slider", 2), ("noise_slider", 1)):
        st.session_state[key] = round(st.session_state.get(key, SETTING_SPECS[key].default) * step) / step
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
