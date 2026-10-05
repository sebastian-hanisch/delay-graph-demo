"""Die Aussagen der App als Tests: jede Zahl in den Hilfetexten, Presets, Tabellen und Szenen ist hier über die festen Sweep-Datensätze belegt (Toleranzen bewusst weit). Positive UND negative Aussagen.
Der Vergleich ist ein eigener Nachbau an synthetischen Daten; die Zahlen sind Mittel über fünf feste Aufnahmen."""

import dataclasses
from functools import lru_cache

import numpy as np
import pytest

import dg_algorithm as alg
import dg_constants as C
import dg_evaluation as ev


def _slim(analysis):
    """Die Tests lesen nur Kennzahlen (und `ev.verdict` aus dem Datensatz nur params und n_electrodes). Eine volle Analyse belegt ~134 MiB
    (Signale des Datensatzes, Matching-Residuen, Verzögerungsdaten), der Cache hält dutzende Varianten mal fünf Aufnahmen - auf dem
    CI-Rechner wurde der Lauf wegen Speichermangels beendet (Exit 137)."""
    return dataclasses.replace(analysis, ds=dataclasses.replace(analysis.ds, X=None, X_clean=None, S=None), delay=None, matching=None)


@lru_cache(maxsize=None)
def _cached(items, settings, names):
    return tuple(_slim(ev.analyse(ev.make_dataset(seed=s, **dict(items)), settings, comparators_used=names)) for s in C.SWEEP_SEEDS)


def _A(settings=ev.Settings(), names=("ica",), **kw):
    return _cached(tuple(sorted(kw.items())), settings, names)


def _f1(analyses, method):
    if method == "dg":
        return float(np.mean([a.dg.f1 for a in analyses]))
    if method == "pipe":
        return float(np.mean([a.pipe.f1 for a in analyses]))
    if method == "tm":
        return float(np.mean([a.tm.f1 for a in analyses]))
    return float(np.mean([a.comparators[method]["f1"] for a in analyses]))


def _near(value, expected, tol=0.05):
    return abs(value - expected) <= tol


# --- Standardfall ------------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_default_scene_numbers():
    """Beleg für 'Standardfall': Verzögerungsgraph 0.97 gegen Pipeline 0.73, Abgleich 0.85, ICA 0.72; trifft 99 %, ordnet 98 % richtig zu; Nachbarschaftsfehler klein; 20 % Einzelkontakt-Zellen."""
    a = _A()
    assert _near(_f1(a, "dg"), 0.97, 0.03) and _near(_f1(a, "pipe"), 0.73, 0.05) and _near(_f1(a, "tm"), 0.85, 0.05) and _near(_f1(a, "ica"), 0.72, 0.06)
    assert _near(np.mean([x.dg.recall for x in a]), 0.99, 0.02) and _near(np.mean([x.dg.accuracy for x in a]), 0.98, 0.02) and np.mean([x.dg.precision for x in a]) > 0.96
    assert np.mean([x.eta_m for x in a]) < 0.08 and np.mean([x.eta_f for x in a]) < 0.08 and _near(np.mean([x.single_contact for x in a]), 0.20, 0.06)
    assert all(x.ambiguous_pairs == 0 for x in a)


def test_pipeline_needs_more_than_three_components_for_a_dozen_cells():
    """Beleg für die Erläuterung zur Pipeline: bei drei Hauptkomponenten 0.62, mit der Zellzahl als Komponentenzahl 0.73."""
    three = []
    for s in C.SWEEP_SEEDS:
        ds = ev.make_dataset(seed=s)
        srt = alg.sort_spikes(ds.X, ds.n_neurons, n_components=3, cluster_mode="known", seed=1)
        three.append(ev.evaluate_events(ds, srt.times, srt.clustering.labels, srt.k).f1)
    assert _near(np.mean(three), 0.62, 0.05) and _f1(_A(), "pipe") > np.mean(three) + 0.06
    assert ev.pipeline_components(ev.make_dataset(m=12)) == 12 and ev.pipeline_components(ev.make_dataset(m=4)) == 4 and ev.pipeline_components(ev.make_dataset(m=30)) == 15


def test_the_delay_graph_is_much_faster_than_pipeline_plus_matching():
    a = _A()
    assert np.mean([x.seconds["dg"] * 5 for x in a]) < np.mean([x.seconds["matching"] for x in a])


@pytest.mark.slow
def test_excluding_single_contact_cells_costs_recall_in_the_default_scene():
    """Beleg für die Hilfe zu den Einzelkontakt-Zellen: ausgeschlossen 0.77 im Standardfall (zugelassen 0.97)."""
    allowed, excluded = _A(), _A(ev.Settings(min_clique=2))
    assert _near(_f1(excluded, "dg"), 0.77, 0.05) and _f1(allowed, "dg") > _f1(excluded, "dg") + 0.15


# --- Sidebar-Hilfen: Zellen, Gitter, Kontakte ----------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("m,dg,pipe,tm,ica", [(6, 0.99, 0.93, 0.95, 0.85), (25, 0.95, 0.52, 0.76, 0.55), (30, 0.94, 0.47, 0.74, 0.50)])
def test_cell_count_help_text(m, dg, pipe, tm, ica):
    a = _A(m=m)
    assert _near(_f1(a, "dg"), dg, 0.04) and _near(_f1(a, "pipe"), pipe, 0.06) and _near(_f1(a, "tm"), tm, 0.06) and _near(_f1(a, "ica"), ica, 0.07)


@pytest.mark.slow
@pytest.mark.parametrize("grid,dg,pipe,tm,ica", [(3, 0.86, 0.76, 0.92, 0.42), (4, 0.91, 0.74, 0.82, 0.64), (6, 0.99, 0.78, 0.85, 0.79)])
def test_grid_help_text(grid, dg, pipe, tm, ica):
    a = _A(grid=grid)
    assert _near(_f1(a, "dg"), dg, 0.05) and _near(_f1(a, "pipe"), pipe, 0.06) and _near(_f1(a, "tm"), tm, 0.06) and _near(_f1(a, "ica"), ica, 0.08)
    if grid == 3:
        assert _f1(a, "tm") > _f1(a, "dg")                                                    # auf dem kleinen Gitter gewinnt der Abgleich


@pytest.mark.slow
def test_contact_probability_help_text():
    """Beleg: p = 0.1 -> 63 % Einzelkontakt-Zellen, F1 0.86 (Pipeline 0.71, Abgleich 0.74, ICA 0.71), ausgeschlossen 0.37 (Trefferquote 0.36, Sortiergenauigkeit 0.98); ab 0.35 nahe 1 (0.97 / 0.98)."""
    low = _A(contact_p=0.1)
    assert _near(np.mean([x.single_contact for x in low]), 0.63, 0.06) and _near(_f1(low, "dg"), 0.86, 0.05) and _near(_f1(low, "pipe"), 0.71, 0.06) and _near(_f1(low, "tm"), 0.74, 0.06) and _near(_f1(low, "ica"), 0.71, 0.07)
    ex = _A(ev.Settings(min_clique=2), contact_p=0.1)
    assert _near(_f1(ex, "dg"), 0.37, 0.06) and _near(np.mean([x.dg.recall for x in ex]), 0.36, 0.06) and _near(np.mean([x.dg.accuracy for x in ex]), 0.98, 0.03)
    assert _near(_f1(_A(contact_p=0.35), "dg"), 0.97, 0.03) and _near(_f1(_A(contact_p=0.5), "dg"), 0.98, 0.03)


@pytest.mark.slow
@pytest.mark.parametrize("similarity,dg,pipe,tm", [(0.0, 0.99, 0.67, 0.81), (1.0, 0.97, 0.73, 0.85)])
def test_similarity_help_text(similarity, dg, pipe, tm):
    a = _A(similarity=similarity)
    assert _near(_f1(a, "dg"), dg, 0.03) and _near(_f1(a, "pipe"), pipe, 0.06) and _near(_f1(a, "tm"), tm, 0.06)


@pytest.mark.slow
@pytest.mark.parametrize("delay,ica", [(0, 0.77), (8, 0.62)])
def test_delay_range_help_text(delay, ica):
    a = _A(delay_max=delay)
    assert _near(_f1(a, "dg"), 0.99, 0.03) and _near(_f1(a, "ica"), ica, 0.07)


# --- Sidebar-Hilfen: Jitter, Feuerrate, Rauschen, Länge ----------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("jitter,dg,tm", [(1.0, 0.91, 0.80), (2.0, 0.70, 0.77), (3.0, 0.62, 0.63)])
def test_jitter_help_text(jitter, dg, tm):
    a = _A(jitter=jitter)
    assert _near(_f1(a, "dg"), dg, 0.05) and _near(_f1(a, "tm"), tm, 0.06)
    if jitter == 2.0:
        assert _f1(a, "tm") > _f1(a, "dg")                                                    # bei Jitter 2 gewinnt der Abgleich


@pytest.mark.slow
def test_low_firing_rate_is_a_weakness_and_high_firing_rate_a_strength():
    """Beleg: Faktor 0.25 (unter der angenommenen Untergrenze): 0.72 gegen Pipeline 0.86 und Abgleich 0.82, Genauigkeit der Detektion 0.62, Nachbarschaften 57 % fehlend / 48 % falsch; Faktor 4: 0.96 bei Pipeline 0.48."""
    low = _A(rate_scale=0.25)
    assert _near(_f1(low, "dg"), 0.72, 0.06) and _near(_f1(low, "pipe"), 0.86, 0.06) and _near(_f1(low, "tm"), 0.82, 0.06) and _f1(low, "pipe") > _f1(low, "dg")
    assert _near(np.mean([x.dg.precision for x in low]), 0.62, 0.08) and _near(np.mean([x.eta_m for x in low]), 0.57, 0.12) and _near(np.mean([x.eta_f for x in low]), 0.48, 0.12)
    high = _A(rate_scale=4.0)
    assert _near(_f1(high, "dg"), 0.96, 0.03) and _near(_f1(high, "pipe"), 0.48, 0.07) and _near(_f1(high, "tm"), 0.82, 0.07) and _near(_f1(high, "ica"), 0.74, 0.08)


@pytest.mark.slow
@pytest.mark.parametrize("noise,dg,pipe,tm,ica", [(20.0, 0.88, 0.58, 0.69, 0.57), (30.0, 0.72, 0.27, 0.36, 0.42), (40.0, 0.60, 0.10, 0.11, 0.22)])
def test_noise_help_text(noise, dg, pipe, tm, ica):
    a = _A(noise=noise)
    assert _near(_f1(a, "dg"), dg, 0.07) and _near(_f1(a, "pipe"), pipe, 0.07) and _near(_f1(a, "tm"), tm, 0.08) and _near(_f1(a, "ica"), ica, 0.08)
    if noise == 20.0:
        assert _near(np.mean([x.eta_m for x in a]), 0.20, 0.08) and _near(np.mean([x.eta_f for x in a]), 0.45, 0.10)        # die Fehlermaße der Dissertation steigen stark, die Zuordnung leidet weniger


def test_one_second_of_recording_is_enough():
    a = _A(seconds=1.0)
    assert _near(_f1(a, "dg"), 0.99, 0.03) and _near(_f1(a, "pipe"), 0.72, 0.06) and _near(_f1(a, "tm"), 0.73, 0.06)


# --- Presets und Szenen ------------------------------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_many_contacts_same_shapes_preset_numbers():
    """Beleg: 20 Zellen, 4×4, p 0.5, Ähnlichkeit 0: Verzögerungsgraph 0.95, Abgleich 0.85, Pipeline 0.55, ICA 0.43, SCA 0.20."""
    a = _A(names=("ica", "sca"), contact_p=0.5, similarity=0.0, m=20, grid=4)
    assert _near(_f1(a, "dg"), 0.95, 0.04) and _near(_f1(a, "tm"), 0.85, 0.06) and _near(_f1(a, "pipe"), 0.55, 0.07) and _near(_f1(a, "ica"), 0.43, 0.08) and _near(_f1(a, "sca"), 0.20, 0.10)


@pytest.mark.slow
def test_more_cells_than_electrodes_preset_numbers():
    """Beleg: 30 Zellen auf 25 Elektroden: 0.94 (Abgleich 0.74, ICA 0.50, Pipeline 0.47), SCA 0.07; im Mittel ein Zellenpaar mit identischer Charakteristik."""
    a = _A(names=("ica", "sca"), m=30)
    assert _near(_f1(a, "dg"), 0.94, 0.04) and _near(_f1(a, "tm"), 0.74, 0.06) and _near(_f1(a, "ica"), 0.50, 0.08) and _near(_f1(a, "pipe"), 0.47, 0.07) and _near(_f1(a, "sca"), 0.07, 0.07)
    assert _near(np.mean([x.ambiguous_pairs for x in a]), 1.0, 0.8)


@pytest.mark.slow
def test_crowded_single_contact_cells_are_a_weakness_because_of_proposition_2_3_1():
    a = _A(m=20, grid=3, contact_p=0.1)
    assert _near(np.mean([x.ambiguous_pairs for x in a]), 8.8, 3.0) and _near(_f1(a, "dg"), 0.59, 0.07) and _near(_f1(a, "tm"), 0.68, 0.07) and _f1(a, "dg") <= max(_f1(a, "tm"), _f1(a, "pipe")) + 0.02


@pytest.mark.slow
@pytest.mark.parametrize("kwargs,settings,codes", [
    ({}, ev.Settings(), {"delay_wins", "comparable", "waveform_wins"}),                      # auf einzelnen Datensätzen holt der Abgleich auf (kleinster F1 des Verzögerungsgraphen 0.89)
    (dict(m=30), ev.Settings(), {"delay_wins", "comparable", "waveform_wins"}),
    (dict(jitter=2.0), ev.Settings(), {"jitter", "waveform_wins", "comparable"}),
    (dict(rate_scale=0.25), ev.Settings(), {"rate", "comparable", "waveform_wins"}),
    (dict(noise=40.0), ev.Settings(), {"noise"}),
    (dict(contact_p=0.1), ev.Settings(min_clique=2), {"singles_excluded"}),
    (dict(m=20, grid=3, contact_p=0.1), ev.Settings(), {"ambiguous", "waveform_wins", "comparable"}),
])
def test_verdict_codes_hold_on_several_datasets(kwargs, settings, codes):
    for a in _A(settings, **kwargs):
        assert ev.verdict(a)[1] in codes, (kwargs, ev.verdict(a)[1])


@pytest.mark.slow
def test_sweep_and_scene_tables_have_the_expected_shape_and_the_scenes_split_into_strengths_and_weaknesses():
    rows = ev.sweep("seconds", values=(1.0, 2.0))
    again = ev.sweep("seconds", values=(1.0, 2.0))
    strip = lambda rs: [{k: v for k, v in r.items() if not k.startswith("sec_")} for r in rs]                # Rechenzeiten schwanken
    assert strip(rows) == strip(again) and [r["x"] for r in rows] == [1.0, 2.0]
    assert all(set(r) >= {"dg", "pipe", "tm", "ica", "recall", "precision", "accuracy", "eta_m", "eta_f", "single_contact", "dg_min", "dg_max", "dg_std"} for r in rows)
    scenes = ev.scene_table()
    assert [r["scene"] for r in scenes] == [s for s, _ in ev.SCENES] and all(r["dg_min"] <= r["dg"] <= r["dg_max"] for r in scenes)
    for r in scenes[:5]:                                                                       # Stärken: der Verzögerungsgraph liegt vorn
        assert r["dg"] > max(r["pipe"], r["tm"], r["ica"]) + 0.03, r["scene"]
    for r in scenes[5:]:                                                                       # Schwächen: ein anderes Verfahren ist besser oder gleich
        assert r["dg"] <= max(r["pipe"], r["tm"], r["ica"]) + 0.02, r["scene"]
    assert all(r["sec_dg"] < r["sec_matching"] for r in scenes)


@pytest.mark.slow
def test_delays_add_little_in_this_model_because_electrode_sets_already_tell_cells_apart():
    """Beleg für die ehrliche Einordnung: ohne Verzögerungen (Spanne 0) erreicht der Verzögerungsgraph 0.99 wie mit ihnen; auch auf einem dicht besetzten kleinen Gitter (20 Zellen, 3×3, p 0.5) kaum Unterschied."""
    assert _near(_f1(_A(delay_max=0), "dg"), _f1(_A(delay_max=8), "dg"), 0.03)
    dense0, dense4 = _A(m=20, grid=3, contact_p=0.5, delay_max=0), _A(m=20, grid=3, contact_p=0.5, delay_max=4)
    assert abs(_f1(dense0, "dg") - _f1(dense4, "dg")) < 0.06


def test_sca_with_the_settings_of_the_sca_demo_finds_nothing_at_25_electrodes_but_would_with_a_looser_threshold():
    """Beleg für die Grenzen-Einordnung: Aktivitätsschwelle 4 x Rausch-Norm -> F1 unter 0.2; Schwelle 2 -> etwa 0.56 (eine Aufnahme, Seed 100000)."""
    import dg_sca as sca
    ds = ev.make_dataset(seed=100000)

    def sca_f1():
        est = sca.fit_sca(ds.X, ds.n_neurons, "l1", seed=1).sources
        idx, corr, aligned = ev.matched(ds.S, est)
        return float(np.mean([ev.spike_f1(ev.detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(ds.n_neurons)]))

    original = sca.ACTIVE_NOISE_FACTOR
    try:
        assert sca_f1() < 0.2
        sca.ACTIVE_NOISE_FACTOR = 2.0
        assert 0.4 < sca_f1() < 0.7
    finally:
        sca.ACTIVE_NOISE_FACTOR = original
