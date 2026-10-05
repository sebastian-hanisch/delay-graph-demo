"""Rauchtests der Streamlit-Oberfläche per AppTest: Standard, jedes Preset, Randgrößen, Schritt-Zustand, Experimente auf Abruf, Achsensperre."""

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dg_constants as C
from dg_presets import PRESET_KEYS

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"


def _run(setup=None, timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if setup is not None:
        setup(at)
        at.run()
        assert not at.exception, [e.value for e in at.exception]
    return at


def _apply(at, p):
    for key, state_key in PRESET_KEYS.items():
        at.session_state[state_key] = p[key]


def test_default_renders_without_exception():
    at = _run()
    assert any("Verzögerungsgraph in Aktion" in m.value for m in at.markdown)
    assert not at.error and not at.warning and len(at.success) == 1 and any("Eigener Nachbau" in i.value for i in at.info)


@pytest.mark.slow
@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_renders(name):
    at = _run(lambda a: _apply(a, C.PRESETS[name]))
    assert len(at.success) + len(at.warning) == 1


@pytest.mark.slow
def test_extreme_settings_render():
    def small(at):
        at.session_state["n_cells_slider"] = C.N_CELLS_MIN
        at.session_state["grid_slider"] = C.GRID_MIN
        at.session_state["seconds_slider"] = C.SECONDS_MIN
        at.session_state["noise_slider"] = C.NOISE_MAX
        at.session_state["jitter_slider"] = C.JITTER_MAX
        at.session_state["contact_p_slider"] = C.CONTACT_P_MIN
    _run(small)

    def large(at):
        at.session_state["n_cells_slider"] = C.N_CELLS_MAX
        at.session_state["grid_slider"] = C.GRID_MAX
        at.session_state["contact_p_slider"] = C.CONTACT_P_MAX
        at.session_state["rate_slider"] = C.RATE_SCALE_MAX
        at.session_state["delay_max_slider"] = C.DELAY_MAX_MAX
        at.session_state["similarity_slider"] = C.SIMILARITY_MIN
        at.session_state["seconds_slider"] = 4.0
        at.session_state["singles_select"] = "excluded"
    _run(large)


def test_almost_no_spikes_render_in_every_step():
    def setup(at):
        at.session_state["rate_slider"] = C.RATE_SCALE_MIN
        at.session_state["n_cells_slider"] = C.N_CELLS_MIN
        at.session_state["seconds_slider"] = C.SECONDS_MIN
        at.session_state["noise_slider"] = C.NOISE_MAX
    for step in (1, 2, 3, 4, 5):
        at = _run(setup)
        at.session_state["dg_step"] = step
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def test_step_state_resets_when_the_data_or_settings_change_and_survives_reruns():
    at = _run()
    at.session_state["dg_step"] = 4
    at.run()
    assert not at.exception and at.session_state["dg_step"] == 4
    at.session_state["noise_slider"] = 12.0
    at.run()
    assert not at.exception and at.session_state["dg_step"] == 1


@pytest.mark.parametrize("step", [1, 2, 3, 4, 5])
def test_every_step_renders(step):
    def setup(at):
        at.session_state["dg_step"] = step
    _run(setup)


def test_window_start_is_clamped_when_the_recording_gets_shorter():
    at = _run(lambda a: a.session_state.__setitem__("window_start", 2000))
    at.session_state["seconds_slider"] = C.SECONDS_MIN
    at.run()
    assert not at.exception and at.session_state["window_start"] == 0


@pytest.mark.slow
def test_sweep_and_scenes_run_on_demand():
    at = _run()
    [s for s in at.selectbox if s.key == "sweep_select"][0].select("seconds")
    at.run()
    [b for b in at.button if b.key == "sweep_start"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    at.session_state["singles_select"] = "excluded"
    at.run()
    assert not at.exception
    [b for b in at.button if b.key == "scenes_start"][0].click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["scenes_on"]


def test_every_figure_of_the_visualisation_module_is_axis_locked():
    source = (ROOT / "dg_visualization.py").read_text(encoding="utf-8")
    assert len(re.findall(r"return lock_axes\(fig\)", source)) == len(re.findall(r"^def build_", source, flags=re.M))
    assert len(re.findall(r"^\s+return fig$", source, flags=re.M)) == 1


def test_every_plotly_chart_has_an_explicit_key():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    calls = re.findall(r"plotly_chart\(", source)
    keyed = re.findall(r"plotly_chart\(.*?key=\"[a-z_]+\"\)", source)
    assert len(calls) == len(keyed) >= 12
