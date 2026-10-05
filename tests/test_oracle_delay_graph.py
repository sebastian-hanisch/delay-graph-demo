"""Orakel (anderer Rechenweg als der eigene Code): Doppelschleifen und Brute Force für Algorithmus 1 und 2, Kantenmenge/Nachbarschaften nach Definition, Gesamtverfahren auf den
exakten Knoten des Modells (Eingabe = wahre Spitzenzeiten; Orakel = die wahren Spikes), Schleifen-Nachbildung des Modells, naiver Matching Pursuit (Korrelation nach jedem Abzug
neu gerechnet), scikit-learn (PCA, Silhouette, k-means), SciPy (Zuordnung). Kleine Instanzen, wenige Sekunden."""

import itertools
import math
import warnings

import numpy as np
import pytest

import dg_algorithm as alg
import dg_constants as C
import dg_delay as dd
import dg_evaluation as ev
import dg_matching as tmm
import dg_scenario as sc

sp_opt = pytest.importorskip("scipy.optimize")
sk_decomp = pytest.importorskip("sklearn.decomposition")
sk_cluster = pytest.importorskip("sklearn.cluster")
sk_metrics = pytest.importorskip("sklearn.metrics")

WINDOW = 4 * C.EPSILON


def test_bounded_differences_equal_a_double_loop():
    rng = np.random.default_rng(1)
    for _ in range(60):
        p1 = np.sort(rng.choice(2000, int(rng.integers(0, 25)), replace=False))
        p2 = np.sort(rng.choice(2000, int(rng.integers(0, 25)), replace=False))
        delta = int(rng.integers(1, 40))
        l1, l2, d = dd.bounded_differences(p1, p2, delta)
        ref = sorted((a, b, int(p2[b] - p1[a])) for a in range(len(p1)) for b in range(len(p2)) if abs(int(p2[b]) - int(p1[a])) <= delta)
        assert sorted(zip(l1.tolist(), l2.tolist(), d.tolist())) == ref


def test_admissible_mask_equals_brute_force_over_all_integer_windows():
    rng = np.random.default_rng(2)
    for _ in range(60):
        n, nu = int(rng.integers(1, 60)), int(rng.integers(3, 30))
        centre = rng.integers(-20, 20, size=int(rng.integers(1, 4)))
        diffs = np.concatenate([rng.choice(centre, n // 2 + 1) + rng.integers(-3, 4, n // 2 + 1), rng.integers(-40, 41, n // 2)]).astype(int)
        need = (1 - C.THETA1) * nu
        ref = np.array([any(np.sum((diffs >= a) & (diffs <= a + WINDOW)) >= need for a in range(x - WINDOW, x + 1)) for x in diffs])
        assert np.array_equal(dd.admissible_mask(diffs, nu), ref)


def _random_peaks(rng, ne, nu, dmax, T=6000):
    pk = [[] for _ in range(ne)]
    for _ in range(int(rng.integers(1, 4))):
        els = np.sort(rng.choice(ne, int(rng.integers(1, min(ne, 3) + 1)), replace=False))
        dl = rng.integers(0, dmax + 1, len(els))
        ts = np.sort(rng.integers(100, T - 100, int(rng.integers(nu, nu + 6))))
        for j, d in zip(els, dl):
            pk[j].extend((ts + d).tolist())
    for j in range(ne):
        pk[j].extend(rng.integers(0, T, int(rng.integers(0, 5))).tolist())
    return [np.array(sorted(set(p)), dtype=int) for p in pk]


def test_graph_neighbourhoods_and_greedy_cliques_equal_their_definitions():
    rng = np.random.default_rng(3)
    for _ in range(12):
        ne, nu, dmax = int(rng.integers(2, 6)), int(rng.integers(3, 12)), int(rng.integers(0, 9))
        peaks = _random_peaks(rng, ne, nu, dmax)
        pos = np.column_stack([0.5 + np.arange(ne), np.full(ne, 0.5)])
        g = dd.build_graph(peaks, pos, nu, dmax)
        delta = dmax + 2 * C.EPSILON
        need = (1 - C.THETA1) * nu
        ref = set()
        for j1 in range(ne):
            for j2 in range(j1 + 1, ne):
                if np.linalg.norm(pos[j1] - pos[j2]) > 2 * C.RADIUS:
                    continue
                pairs = [(a, b, int(peaks[j2][b] - peaks[j1][a])) for a in range(len(peaks[j1])) for b in range(len(peaks[j2])) if abs(int(peaks[j2][b] - peaks[j1][a])) <= delta]
                dif = np.array([d for _, _, d in pairs], dtype=int)
                ref |= {((j1, a), (j2, b)) for a, b, d in pairs if any(np.sum((dif >= s) & (dif <= s + WINDOW)) >= need for s in range(d - WINDOW, d + 1))}
        got = {(node, (j2, l2)) for node, nb in g.adjacency.items() for j2, ls in nb.items() for l2 in ls if node[0] < j2}
        assert got == ref
        counts = {}
        for node in g.adjacency:
            nbset = frozenset([node[0]] + [j2 for j2, ls in g.adjacency[node].items() if ls])
            counts[nbset] = counts.get(nbset, 0) + 1
        confirmed, allnb = dd.estimate_neighbourhoods(g)
        assert allnb == counts and confirmed == {k: v for k, v in counts.items() if v >= (1 - C.THETA2) * nu * len(k)}

        def linked(a, b):
            return b[1] in g.adjacency[a].get(b[0], ())

        for node in list(g.adjacency)[:25]:
            cl = dd.greedy_clique(g, node)
            assert node in cl and all(linked(a, b) for a, b in itertools.combinations(cl, 2))
            assert not any(c not in cl and all(linked(c, x) for x in cl) for c in g.adjacency)         # maximal


@pytest.mark.parametrize("seed", [100000, 100001, 100002])
def test_sorting_on_the_exact_model_peaks_recovers_the_true_spikes(seed):
    """Eingabe = wahre Spitzenzeiten aller Kontakte (ohne Schätzfehler); Orakel = die wahren Spikes. Ohne Rauschen und Jitter muss das Verfahren fast alles finden."""
    ds = sc.make_dataset(8, 4, 0.5, 1.0, 4, 0.0, 1.0, 10.0, 2.0, seed)
    peaks = [np.array(p, dtype=int) for p in ds.electrode_peaks]
    g = dd.build_graph(peaks, ds.positions, ds.nu, 4)
    s = dd.sort_spikes(g, 1)
    r = ev.evaluate_events(ds, s.times, s.labels, max(len(s.candidates), 1))
    assert r.recall > 0.97 and r.precision > 0.97 and r.accuracy > 0.93 and r.f1 > 0.9
    confirmed, _ = dd.estimate_neighbourhoods(g)
    eta_m, eta_f = dd.neighbourhood_errors(confirmed, sc.neighbourhood_true(ds))
    assert eta_m <= 0.2 and eta_f <= 0.2


@pytest.mark.parametrize("m,grid,p,sim,dm,jit", [(4, 3, 0.5, 1.0, 4, 0.0), (5, 4, 0.3, 0.0, 8, 2.0), (3, 3, 0.5, 0.5, 0, 0.0)])
def test_model_equals_a_loop_rebuild_of_the_signal(m, grid, p, sim, dm, jit):
    T_s, seed = 0.5, 9
    ds = sc.make_dataset(m, grid, p, sim, dm, jit, 1.0, 0.0, T_s, seed)
    T = int(round(T_s * C.SAMPLE_RATE))
    pos = [(0.5 + (j % grid), 0.5 + (j // grid)) for j in range(grid * grid)]
    X = np.zeros((grid * grid, T))

    def phi_of(par):
        w = C.WIDTH
        b1, b2 = par["beta1"] * w, 0.5 * w
        phi = [par["alpha1"] * math.exp(-((k - b1) ** 2) / (2 * (par["gamma1"] * w) ** 2)) + par["alpha2"] * math.exp(-((k - b2) ** 2) / (2 * (par["gamma2"] * w) ** 2)) for k in range(w + 1)]
        return phi, int(np.argmin(phi))

    for i in range(m):
        cell, contacts, delays, params, pseudo, pdelays, pparams = sc._cell(i, seed, grid, p, sim, dm)
        assert np.array_equal(contacts, ds.neighbourhoods[i]) and all(math.dist(pos[j], cell) <= C.RADIUS + 1e-9 for j in contacts)
        s = ds.firing_times[i]
        jrng = np.random.default_rng([seed, 4000 + i])
        for c, (j, d, par) in enumerate(zip(contacts, delays, params)):
            jt = np.rint(jit * jrng.standard_normal(len(s))).astype(int) if jit > 0 else np.zeros(len(s), int)
            phi, pi = phi_of(par)
            for sk_, jk in zip(s, jt):
                pk = int(sk_ + d + jk)
                for k in range(C.WIDTH + 1):
                    if 0 <= pk - pi + k < T:
                        X[j, pk - pi + k] += phi[k]
        for j, d, par in zip(pseudo, pdelays, pparams):
            phi, pi = phi_of(par)
            for sk_ in s:
                for k in range(C.WIDTH + 1):
                    if 0 <= int(sk_ + d) - pi + k < T:
                        X[j, int(sk_ + d) - pi + k] += C.BACKGROUND_ATTENUATION * phi[k]
        assert np.array_equal(ds.spike_times[i], s + int(delays.min()))
        assert ds.characteristics[i] == tuple((int(j), int(d - delays[0])) for j, d in zip(contacts, delays))
    assert np.allclose(X, ds.X_clean, atol=1e-6)


def test_truth_spikes_collision_flag_equals_a_pairwise_loop():
    ds = sc.make_dataset(7, 4, 0.3, 1.0, 4, 0.0, 4.0, 10.0, 0.5, 3)
    tt, tn, tc = ev.truth_spikes(ds)
    order = np.argsort(np.concatenate(ds.spike_times), kind="stable")
    fire = np.concatenate(ds.firing_times)[order]
    ref = np.array([sum(1 for f2 in fire if abs(int(f2) - int(f1)) <= C.COLLISION_WINDOW) > 1 for f1 in fire])
    assert np.array_equal(tc, ref) and np.array_equal(tt, np.concatenate(ds.spike_times)[order])


def test_hungarian_assignment_equals_scipy_on_rectangular_and_tied_matrices():
    rng = np.random.default_rng(4)
    for _ in range(40):
        k, nc = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        A = np.round(rng.random((k, nc)), 1 if rng.random() < 0.4 else 6)
        idx = ev.assign(A)
        r, c = sp_opt.linear_sum_assignment(A, maximize=True)
        used = [j for j in idx if j >= 0]
        assert len(used) == len(set(used)) and sum(A[i, j] for i, j in enumerate(idx) if j >= 0) == pytest.approx(A[r, c].sum(), abs=1e-9)


def _naive_pursuit(Xw, W, threshold, lo_a, hi_a):
    """Gleiche Auswahlregeln, aber die Korrelationen werden nach jedem Abzug direkt neu berechnet (kein FFT, keine Kreuzkorrelations-Aktualisierung)."""
    L = tmm.L
    R = Xw.copy()
    n, T = R.shape
    S = T - L + 1
    Q = W.shape[0]
    norm = np.sqrt((W ** 2).sum(axis=(1, 2)))
    rejected = np.zeros((Q, S), bool)
    events = []
    while len(events) < int(tmm.MAX_EVENTS_FACTOR * T) + 10:
        Z = np.array([sum(np.correlate(R[j], W[q, j], mode="valid") for j in range(n)) / norm[q] for q in range(Q)])
        Z[rejected] = -np.inf
        q0, s = divmod(int(np.argmax(Z)), S)
        if Z[q0, s] < threshold:
            break
        a_raw = Z[q0, s] / norm[q0]
        if a_raw < lo_a:
            rejected[q0, max(0, s - tmm.REJECT_RADIUS): s + tmm.REJECT_RADIUS + 1] = True
            continue
        a = min(a_raw, hi_a)
        events.append((s, q0))
        R[:, s: s + L] -= a * W[q0]
        rejected[q0, s] = True
    return sorted(events), R


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matching_pursuit_equals_a_naive_pursuit_with_direct_correlations(seed):
    rng = np.random.default_rng(seed)
    L = tmm.L
    n, Q, T = 2, 2, 320
    W = rng.standard_normal((Q, n, L)) * np.exp(-((np.arange(L) - 30) / 8.0) ** 2)
    X = rng.standard_normal((n, T))
    for _ in range(3):
        s = int(rng.integers(0, T - L))
        X[:, s: s + L] += rng.uniform(0.6, 1.4) * 3 * W[int(rng.integers(0, Q))]
    p = tmm.matching_pursuit(X, W, 5.0, 0.5, 1.5)
    ref_events, ref_res = _naive_pursuit(X, W, 5.0, 0.5, 1.5)
    assert sorted(zip((p.times - C.SNIPPET_BEFORE).tolist(), p.templates.tolist())) == ref_events and np.allclose(p.residual, ref_res, atol=1e-8)


def test_pipeline_building_blocks_equal_scikit_learn_and_loops():
    rng = np.random.default_rng(5)
    for _ in range(10):
        N, n, L, k = int(rng.integers(30, 100)), int(rng.integers(1, 4)), int(rng.integers(8, 25)), int(rng.integers(1, 5))
        sn = rng.standard_normal((N, n, L)) + 2 * rng.standard_normal((1, n, L))
        f = alg.pca_features(sn, k)
        ref = sk_decomp.PCA(n_components=f.values.shape[1], svd_solver="full").fit(sn.reshape(N, -1))
        assert np.allclose(np.abs(f.values), np.abs(ref.transform(sn.reshape(N, -1))), atol=1e-7)
        assert np.allclose(f.explained, ref.explained_variance_ratio_, atol=1e-9)
    for t in range(5):
        kk, N = int(rng.integers(2, 5)), int(rng.integers(40, 120))
        F = rng.normal(size=(kk, 3))[rng.integers(0, kk, N)] * 4 + rng.normal(size=(N, 3))
        km = alg.kmeans(F, kk, seed=t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = sk_cluster.KMeans(n_clusters=kk, n_init=20, random_state=0).fit(F)
        assert km.inertia <= sk.inertia_ * 1.02 + 1e-9
        if len(set(km.labels)) > 1:
            assert alg.silhouette(F, km.labels) == pytest.approx(sk_metrics.silhouette_score(F, km.labels), abs=1e-9)
    X = rng.standard_normal((3, 2000)) * rng.uniform(0.5, 3, (3, 1))
    for s in rng.integers(100, 1900, 5):
        X[:, s] -= rng.uniform(10, 20)
    times, d = alg.detect(X, 4.5)
    med, sig = np.median(X, axis=1), alg.noise_sigma(X)
    dref = np.array([min((X[j, i] - med[j]) / sig[j] for j in range(3)) for i in range(2000)])
    assert np.allclose(d, dref) and all(dref[t] < -4.5 for t in times) and (len(times) < 2 or np.diff(times).min() > C.DEAD_TIME)
    assert all(any(abs(c - t) <= C.DEAD_TIME and dref[t] <= dref[c] for t in times) for c in np.flatnonzero(dref < -4.5))


def test_fastica_copy_equals_scikit_learn_with_the_same_start():
    import dg_ica as ica
    rng = np.random.default_rng(12)
    X = rng.standard_normal((4, 4)) @ rng.laplace(size=(4, 2500))
    ours = ica.fit_ica(X, 4, "logcosh", "symmetric", init_start=3, max_iter=500, tol=1e-9)
    W0 = np.random.default_rng([3, 4242]).standard_normal((4, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sk_decomp.FastICA(algorithm="parallel", whiten=False, fun="logcosh", w_init=W0, tol=1e-9, max_iter=500).fit(ours.whitening.Z.T)
    assert np.allclose(ours.W, ref.components_, atol=1e-8)
