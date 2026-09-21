"""Vorlagenabgleich: Handinstanzen (Passung, Kreuzkorrelation, Abziehen, Amplitudengrenzen), Verfeinerung und die Ausrichtung der Start-Vorlagen."""

import numpy as np
import pytest

import dg_constants as C
import dg_matching as tmm

L = tmm.L


def _templates(q=2, n=3, seed=0, scale=3.0):
    return np.random.default_rng(seed).standard_normal((q, n, L)) * scale


def _signal(W, events, T=500):
    X = np.zeros((W.shape[1], T))
    for s, q, a in events:
        X[:, s: s + L] += a * W[q]
    return X


def test_whiten_gives_median_zero_and_unit_noise_and_ignores_spikes():
    rng = np.random.default_rng(1)
    X = 5.0 + np.array([[2.0], [0.5]]) * rng.standard_normal((2, 20000))
    X[0, 100:130] -= 60.0                                                            # ein Spike beeinflusst die robuste Schätzung kaum
    Xw, med, sigma = tmm.whiten(X)
    assert np.allclose(np.median(Xw, axis=1), 0.0, atol=1e-12) and np.allclose(sigma, [2.0, 0.5], rtol=0.03)
    assert np.allclose(np.median(np.abs(Xw), axis=1) / 0.6745, 1.0, rtol=0.03)


def test_correlation_traces_match_a_brute_force_computation():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((3, 400))
    W = _templates()
    Z = tmm.correlation_traces(X, W)
    assert Z.shape == (2, 400 - L + 1)
    for q in (0, 1):
        for s in (0, 17, 100, 300):
            assert Z[q, s] == pytest.approx((X[:, s: s + L] * W[q]).sum() / np.sqrt((W[q] ** 2).sum()))


def test_correlation_of_pure_noise_is_standard_normal():
    """Beleg für 'unter reinem Rauschen ungefähr standardnormalverteilt' (Erwartungswert 0, Standardabweichung 1)."""
    Z = tmm.correlation_traces(np.random.default_rng(3).standard_normal((3, 30000)), _templates())
    assert abs(Z.mean()) < 0.05 and abs(Z.std() - 1.0) < 0.05


def test_cross_correlations_match_a_brute_force_computation():
    W = _templates(q=3)
    XC = tmm.cross_correlations(W)
    for d in (-29, -5, 0, 7, 29):
        for q, p in ((0, 1), (2, 2), (1, 0)):
            expected = sum((W[q, :, l] * W[p, :, l + d]).sum() for l in range(L) if 0 <= l + d < L)
            assert XC[q, p, d + L - 1] == pytest.approx(expected)
    assert XC[1, 1, L - 1] == pytest.approx((W[1] ** 2).sum())                         # Verschiebung 0 = Energie


def test_isolated_events_are_recovered_exactly_and_the_residual_vanishes():
    W = _templates()
    X = _signal(W, [(100, 0, 1.0), (300, 1, 1.2)])
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert list(p.times - C.SNIPPET_BEFORE) == [100, 300] and list(p.templates) == [0, 1] and np.allclose(p.amplitudes, [1.0, 1.2])
    assert np.abs(p.residual).max() < 1e-9


def test_overlapping_events_are_resolved_with_the_right_times_and_templates():
    W = _templates(seed=4)
    X = _signal(W, [(100, 0, 1.0), (112, 1, 0.9)])
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert list(p.times - C.SNIPPET_BEFORE) == [100, 112] and list(p.templates) == [0, 1]
    assert np.allclose(p.amplitudes, [1.0, 0.9], atol=0.15) and np.abs(p.residual).max() < 0.3 * np.abs(X).max()      # gieriges Abziehen ist bei Überlappung nur näherungsweise exakt


def test_the_pursuit_updates_its_scores_exactly_like_a_full_recomputation():
    """Der Kern der Beschleunigung: die Aktualisierung über Kreuzkorrelationen entspricht der neu berechneten Passung des Residuums (kein Rest oberhalb der Schwelle)."""
    W = _templates(seed=5)
    rng = np.random.default_rng(6)
    X = _signal(W, [(60, 0, 1.0), (75, 1, 1.1), (200, 1, 0.8), (210, 0, 1.3), (350, 0, 1.0)]) + 0.02 * rng.standard_normal((3, 500))
    p = tmm.matching_pursuit(X, W, threshold=5.0)
    assert len(p.times) == 5
    assert tmm.correlation_traces(p.residual, W).max() < 5.0


def test_amplitudes_below_the_minimum_are_not_events_and_above_the_maximum_are_clipped():
    W = _templates()
    X = _signal(W, [(100, 0, 0.3), (300, 1, 3.0)])
    assert list(tmm.matching_pursuit(X, W, 5.0, min_amplitude=0.5).templates) == [1]
    low = tmm.matching_pursuit(X, W, 5.0, min_amplitude=0.2)
    assert 100 in (low.times - C.SNIPPET_BEFORE) and 300 in (low.times - C.SNIPPET_BEFORE)
    assert low.amplitudes[list(low.times - C.SNIPPET_BEFORE).index(300)] == pytest.approx(tmm.MAX_AMPLITUDE)          # begrenzt; der Rest darüber wird als weitere Ereignisse abgezogen, die Schleife endet


def _waveform(sigma):
    t = np.arange(L)
    return -np.exp(-(((t - 8) / sigma) ** 2)) + 0.4 * np.exp(-(((t - (8 + 2.5 * sigma)) / (1.6 * sigma)) ** 2))


def test_without_a_minimum_amplitude_leftovers_become_ghost_events():
    """Beleg für 'Ohne Amplitudengrenze': eine um 20 % zu breite Vorlage lässt einen Rest übrig, der ohne Grenze bei kleinem Rauschen als weiterer Spike gemeldet wird (Beleg an den Daten: test_scenario_and_evaluation)."""
    W = np.stack([np.tile(_waveform(3.0) * 30, (3, 1)), np.tile(_waveform(2.0) * 25, (3, 1))])
    X = np.zeros((3, 500))
    X[:, 100: 100 + L] += W[0]
    Wm = W.copy()
    Wm[0] = np.tile(_waveform(3.6) * 30, (3, 1))
    clean = tmm.matching_pursuit(X, Wm, 5.0, min_amplitude=0.5)
    ghosts = tmm.matching_pursuit(X, Wm, 5.0, min_amplitude=0.0)
    assert len(clean.times) == 1 and clean.times[0] - C.SNIPPET_BEFORE == 100 and len(ghosts.times) >= 3


def test_no_templates_or_a_short_signal_give_no_events():
    X = np.random.default_rng(8).standard_normal((3, 400))
    assert len(tmm.matching_pursuit(X, np.zeros((0, 3, L)), 5.0).times) == 0
    assert len(tmm.matching_pursuit(X[:, :10], _templates(), 5.0).times) == 0
    assert len(tmm.matching_pursuit(X, np.zeros((1, 3, L)), 5.0).times) == 0            # Vorlage aus Nullen: nichts passt


def test_cleaned_snippets_of_isolated_events_equal_the_template():
    W = _templates()
    X = _signal(W, [(100, 0, 1.0), (300, 1, 1.2)])
    p = tmm.matching_pursuit(X, W, 5.0)
    cleaned = tmm.cleaned_snippets(W, p)
    assert np.allclose(cleaned[0], W[0]) and np.allclose(cleaned[1], W[1])


