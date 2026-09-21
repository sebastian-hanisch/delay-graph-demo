"""Verzögerungsgraph: Modell der Dissertation (Kontakte, Verzögerungen, Vorlage), Spitzenschätzung, Algorithmen 1 und 2 (Differenzen, zulässige Differenzen), Nachbarschaften, Greedy-Clique,
Charakteristiken und die Grenze aus Proposition 2.3.1 (Zellen mit gleicher normierter Charakteristik sind nicht unterscheidbar)."""

import itertools

import numpy as np
import pytest

import dg_constants as C
import dg_delay as dd
import dg_evaluation as ev
import dg_matching as tmm
import dg_scenario as sc


# --- Szenario / Modell ---------------------------------------------------------------------------------------------------------------


def test_template_has_its_minimum_at_the_anchor_and_stays_near_zero_at_the_borders():
    params = dict(alpha1=-100.0, alpha2=40.0, beta1=0.4625, gamma1=0.0375, gamma2=0.15)
    phi, pi = sc.template(params)
    assert len(phi) == C.WIDTH + 1 and phi[pi] == phi.min() and abs(pi - 0.4625 * C.WIDTH) <= 3 and abs(phi[0]) < 5 and abs(phi[-1]) < 5
    assert phi.min() < -60                                                                 # Spitze (~alpha1 + alpha2 * Anteil)


def test_a_planted_spike_peaks_exactly_at_firing_time_plus_delay():
    ds = ev.make_dataset(m=4, noise=0.0, jitter=0.0, seed=3)
    for i in range(ds.n_neurons):
        for c, j in enumerate(ds.neighbourhoods[i]):
            s = int(ds.firing_times[i][2])
            lo, hi = s + int(ds.delays[i][c]) - 6, s + int(ds.delays[i][c]) + 7
            others = [k for k in range(ds.n_neurons) if k != i and any(abs(int(t) - s) < 100 for t in ds.firing_times[k])]
            if others:
                continue
            assert abs(int(np.argmin(ds.X_clean[j, lo:hi])) + lo - (s + int(ds.delays[i][c]))) <= 1     # das Signal der Zelle dominiert die 5-%-Hintergrundanteile


def test_contacts_are_within_the_radius_and_every_cell_has_a_contact():
    ds = ev.make_dataset(m=30, grid=6, seed=11)
    for i in range(ds.n_neurons):
        assert len(ds.neighbourhoods[i]) >= 1
        dist = np.linalg.norm(ds.positions[ds.neighbourhoods[i]] - ds.cell_positions[i], axis=1)
        assert (dist <= C.RADIUS + 1e-9).all()
        assert (ds.delays[i] >= 0).all() and (ds.delays[i] <= ds.delay_max).all()


def test_mean_cell_degree_is_about_p_pi_r_squared():
    """Dissertation: der Durchschnittsgrad einer Zelle liegt bei ungefähr p*pi*R^2 (hier 2.5; Randeffekte und das Verwerfen von Zellen ohne Kontakt verschieben ihn leicht)."""
    degrees = [len(n) for seed in (1, 2, 3) for n in ev.make_dataset(m=30, grid=6, seed=seed).neighbourhoods]
    assert 1.9 < np.mean(degrees) < 3.1


def test_cells_are_reproducible_and_independent_of_the_cell_count():
    a, b = ev.make_dataset(m=4, seed=5), ev.make_dataset(m=8, seed=5)
    assert (a.X_clean == ev.make_dataset(m=4, seed=5).X_clean).all()
    for i in range(4):
        assert (a.neighbourhoods[i] == b.neighbourhoods[i]).all() and (a.delays[i] == b.delays[i]).all() and (a.firing_times[i] == b.firing_times[i]).all()


def test_zero_similarity_gives_all_contacts_the_same_shape_and_zero_delay_range_gives_no_delays():
    ds = ev.make_dataset(m=6, similarity=0.0, delay_max=0, noise=0.0, seed=2)
    assert all((d == 0).all() for d in ds.delays)
    cell = ds.neighbourhoods[0]
    assert len(set(ds.characteristics[0])) == len(cell) and all(r == 0 for _, r in ds.characteristics[0])


def test_jitter_changes_the_realised_peak_times_but_not_the_firing_times():
    a, b = ev.make_dataset(m=4, jitter=0.0, seed=4), ev.make_dataset(m=4, jitter=2.0, seed=4)
    assert (a.firing_times[0] == b.firing_times[0]).all() and not (a.peak_times[0] == b.peak_times[0]).all() and abs((b.peak_times[0] - a.peak_times[0]).std() - 2.0) < 0.5


def test_firing_rates_are_in_the_configured_range_and_respect_the_refractory_time():
    ds = ev.make_dataset(m=10, seconds=6.0, seed=6)
    for s in ds.firing_times:
        assert (np.diff(s) >= C.REFRACTORY).all() and C.RATE_RANGE[0] * 0.5 <= len(s) / 6.0 <= C.RATE_RANGE[1] * 1.6


# --- Spitzenschätzung ----------------------------------------------------------------------------------------------------------------


def test_peak_estimation_finds_planted_spikes_within_epsilon_and_nothing_in_pure_noise():
    ds = ev.make_dataset(m=3, noise=10.0, seed=8)
    found = 0
    for j in range(ds.n_electrodes):
        est = dd.estimate_peaks(ds.X[j])
        true = ds.electrode_peaks[j]
        if len(true) == 0:
            assert len(est) <= 8                                                          # Elektroden ohne Kontakt: nur Hintergrund (5 %) und Rauschen
            continue
        for t in true:
            near = est[np.abs(est - t) <= 3]
            found += len(near) >= 1
        assert len(est) <= len(true) * 1.1 + 2
    assert found >= 0.9 * sum(len(p) for p in ds.electrode_peaks)
    assert len(dd.estimate_peaks(10.0 * np.random.default_rng(1).standard_normal(40000))) <= 3


def test_peak_estimation_error_is_at_most_a_few_samples():
    ds = ev.make_dataset(m=3, noise=10.0, seed=9)
    errors = []
    for j in range(ds.n_electrodes):
        true = ds.electrode_peaks[j]
        est = dd.estimate_peaks(ds.X[j])
        for t in true:
            near = est[np.abs(est - t) <= 6]
            if len(near):
                errors.append(int(near[np.argmin(np.abs(near - t))] - t))
    assert len(errors) > 20 and np.abs(errors).max() <= 4 and np.mean(np.abs(errors)) < 2


# --- Algorithmus 1 und 2 -----------------------------------------------------------------------------------------------------------


def test_bounded_differences_hand_instance():
    l1, l2, d = dd.bounded_differences(np.array([10, 50, 90]), np.array([12, 55, 200]), 6)
    assert sorted(zip(l1.tolist(), l2.tolist(), d.tolist())) == [(0, 0, 2), (1, 1, 5)]
    assert dd.bounded_differences(np.zeros(0, dtype=int), np.array([1]), 5)[2].size == 0


def test_admissible_differences_are_those_in_a_frequent_window():
    diffs = np.array([5] * 10 + [6] * 2 + [30] * 3 + [100], dtype=int)
    mask = dd.admissible_mask(diffs, nu=10)                                             # Fenster der Länge 4: 5 und 6 zusammen 12 >= 9
    assert mask[:12].all() and not mask[12:].any()
    assert not dd.admissible_mask(np.array([5] * 5), nu=10).any() and dd.admissible_mask(np.zeros(0, dtype=int), 3).size == 0
    spread = np.array([0, 3, 6, 9, 12] * 3)                                              # jedes Fenster von 4 enthält nur 2 Werte -> keine Häufung
    assert not dd.admissible_mask(spread, nu=10).any()


def _planted(cells, T=200_000, seed=0):
    """Spitzenlisten aus geplanten Zellen: cells = [(Elektroden, Verzögerungen, Feuerzeiten)] plus zufällige Störspitzen je Elektrode."""
    n = max(max(e) for e, _, _ in cells) + 1
    peaks = [[] for _ in range(n)]
    for electrodes, delays, times in cells:
        for j, d in zip(electrodes, delays):
            peaks[j].extend(int(t + d) for t in times)
    rng = np.random.default_rng(seed)
    for j in range(n):
        peaks[j].extend(rng.integers(0, T, 6).tolist())
    return [np.array(sorted(set(p)), dtype=int) for p in peaks]


def _positions(n):
    return np.column_stack([0.5 + np.arange(n), np.full(n, 0.5)])


def test_delay_graph_finds_the_edges_of_a_planted_cell_and_the_neighbourhood():
    rng = np.random.default_rng(3)
    times = np.sort(rng.integers(500, 190_000, 40))
    peaks = _planted([((0, 1, 2), (0, 3, 1), times)])
    g = dd.build_graph(peaks, _positions(3), nu=30, delay_max=4)
    confirmed, _ = dd.estimate_neighbourhoods(g)
    assert frozenset({0, 1, 2}) in confirmed and confirmed[frozenset({0, 1, 2})] >= 0.8 * 30 * 3
    assert dd.neighbourhood_errors(confirmed, {frozenset({0, 1, 2})})[0] == 0.0
    l0 = int(np.searchsorted(peaks[0], times[0]))
    assert set(g.adjacency[(0, l0)]) == {1, 2}                                        # das Signal der ersten Feuerung ist mit beiden anderen Elektroden verbunden


def test_greedy_clique_and_characteristic_of_a_planted_instance():
    times = np.array([1000, 5000, 9000, 13000, 17000])
    peaks = _planted([((0, 1, 2), (0, 3, 1), times)], T=20_000)
    g = dd.build_graph(peaks, _positions(3), nu=4, delay_max=4)
    node = (1, int(np.searchsorted(peaks[1], 5003)))
    clique = dd.greedy_clique(g, node)
    electrodes, vec, nodes = dd.characteristic(g, clique)
    assert electrodes == (0, 1, 2) and list(vec) == [3, 1]                            # relative Verzögerungen zur Elektrode mit kleinstem Index
    assert sorted(j for j, _ in clique) == [0, 1, 2]


def test_two_cells_with_the_same_normalised_characteristic_cannot_be_told_apart():
    """Proposition 2.3.1 der Dissertation: gleiche relative Verzögerungen -> nicht unterscheidbar. Zwei Zellen, deren absolute Verzögerungen sich nur um eine Konstante unterscheiden, ergeben einen Kandidaten."""
    rng = np.random.default_rng(5)
    a, b = np.sort(rng.integers(500, 190_000, 30)), np.sort(rng.integers(500, 190_000, 30))
    peaks = _planted([((0, 1), (0, 3), a), ((0, 1), (2, 5), b)])
    g = dd.build_graph(peaks, _positions(2), nu=25, delay_max=4)
    s = dd.sort_spikes(g, min_clique=2)
    assert len(s.candidates) == 1 and s.candidates[0][2] >= 50


def test_two_cells_with_different_relative_delays_are_two_candidates():
    rng = np.random.default_rng(6)
    a, b = np.sort(rng.integers(500, 190_000, 30)), np.sort(rng.integers(500, 190_000, 30))
    peaks = _planted([((0, 1), (0, 3), a), ((0, 1), (0, 0), b)])
    g = dd.build_graph(peaks, _positions(2), nu=25, delay_max=4)
    s = dd.sort_spikes(g, min_clique=2)
    assert len(s.candidates) == 2 and set(int(v[0]) for _, v, _ in s.candidates) == {3, 0}


def test_isolated_peaks_become_neurons_only_with_min_clique_one():
    rng = np.random.default_rng(7)
    times = np.sort(rng.integers(500, 190_000, 30))
    peaks = _planted([((0,), (0,), times)])
    g = dd.build_graph(peaks, _positions(1), nu=25, delay_max=4)
    assert len(dd.sort_spikes(g, min_clique=2).times) == 0
    one = dd.sort_spikes(g, min_clique=1)
    assert len(one.candidates) == 1 and len(one.times) >= 30


def test_sorting_events_are_time_sorted_and_labels_index_the_candidates():
    ds = ev.make_dataset(m=6, seed=12)
    r = dd.run_delay_graph(ds.X, ds.positions, ds.nu, ds.delay_max)
    s = r.sorting
    assert (np.diff(s.times) >= 0).all() and len(s.times) == len(s.labels) == len(s.cliques) and (len(s.labels) == 0 or (s.labels.min() >= 0 and s.labels.max() < len(s.candidates)))


# --- Nachbarschaften und Fehlermaße ---------------------------------------------------------------------------------------------------


def test_neighbourhood_errors_hand_instance():
    true = {frozenset({0, 1}), frozenset({2, 3}), frozenset({4, 5})}
    est = {frozenset({0, 1}), frozenset({2, 3}), frozenset({7})}
    assert dd.neighbourhood_errors(est, true) == (pytest.approx(1 / 3), pytest.approx(1 / 3))


def test_neighbourhoods_of_the_default_scenario_are_recovered():
    for seed in (1, 2, 3):
        ds = ev.make_dataset(seed=seed)
        r = dd.run_delay_graph(ds.X, ds.positions, ds.nu, ds.delay_max)
        eta_m, eta_f = dd.neighbourhood_errors(r.neighbourhoods, sc.neighbourhood_true(ds))
        assert eta_m <= 0.2 and eta_f <= 0.25


# --- Zuordnung (Ungarische Methode) und Abgleich (FFT) ------------------------------------------------------------------------------


def _brute_assign(A):
    k, nc = A.shape
    best, best_val = None, -1.0
    for cols in itertools.product(range(-1, nc), repeat=k):
        used = [c for c in cols if c >= 0]
        if len(used) != len(set(used)):
            continue
        val = sum(A[i, c] for i, c in enumerate(cols) if c >= 0)
        if val > best_val + 1e-12:
            best, best_val = cols, val
    return best_val


@pytest.mark.parametrize("shape", [(3, 3), (2, 4), (4, 2), (4, 4), (1, 3), (3, 1)])
def test_assign_matches_brute_force_optimum(shape):
    rng = np.random.default_rng(shape[0] * 10 + shape[1])
    for _ in range(5):
        A = rng.random(shape)
        idx = ev.assign(A)
        val = sum(A[i, c] for i, c in enumerate(idx) if c >= 0)
        assert val == pytest.approx(_brute_assign(A)) and len([c for c in idx if c >= 0]) == len({c for c in idx if c >= 0})


def test_assign_leaves_zero_gain_rows_unassigned_and_scales_to_thirty():
    A = np.zeros((3, 3))
    A[0, 1] = 1.0
    assert ev.assign(A) == [1, -1, -1]
    big = np.random.default_rng(0).random((30, 30))
    idx = ev.assign(big)
    assert sorted(idx) == list(range(30))


def test_fft_correlation_matches_the_direct_computation():
    rng = np.random.default_rng(4)
    X, W = rng.standard_normal((3, 500)), rng.standard_normal((2, 3, tmm.L))
    Z = tmm.correlation_traces(X, W)
    for q in range(2):
        direct = sum(np.correlate(X[j], W[q, j], mode="valid") for j in range(3)) / np.sqrt((W[q] ** 2).sum())
        assert np.allclose(Z[q], direct, atol=1e-8)


# --- Auswertung ------------------------------------------------------------------------------------------------------------------------


def test_evaluate_events_on_the_true_spikes_is_perfect_and_collisions_are_flagged():
    ds = ev.make_dataset(m=5, rate_scale=2.0, seed=3)
    tt, tn, tc = ev.truth_spikes(ds)
    r = ev.evaluate_events(ds, tt, tn, ds.n_neurons)
    assert r.recall == r.precision == r.accuracy == 1.0 and r.f1 == pytest.approx(1.0) and r.n_ghosts == 0 and tc.any() and (np.diff(tt) >= 0).all()
    e = ev.evaluate_events(ds, np.zeros(0, dtype=int), np.zeros(0, dtype=int), 1)
    assert e.recall == 0.0 and e.f1 == 0.0


def test_single_contact_share_and_indistinguishable_pairs_are_counted():
    ds = ev.make_dataset(m=12, contact_p=0.05, seed=2)
    assert ev.single_contact_share(ds) > 0.5 and ev.indistinguishable_pairs(ds) >= 0


def test_settings_default_follows_the_heuristic_of_the_dissertation():
    assert ev.Settings().min_clique == 1
