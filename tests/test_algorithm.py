import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA
from sklearn.metrics import silhouette_score

import dg_algorithm as alg
import dg_constants as C


def _blobs(k=3, per=80, d=2, spread=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, (k, d)) * 2
    F = np.vstack([c + spread * rng.standard_normal((per, d)) for c in centers])
    return F, np.repeat(np.arange(k), per), centers



# --- PCA ---------------------------------------------------------------------------------------------------------------------------------


def test_pca_features_match_scikit_learn_up_to_sign_and_report_the_explained_variance():
    rng = np.random.default_rng(3)
    snips = rng.standard_normal((200, 3, 30)) @ np.diag(np.linspace(2.0, 0.2, 30))
    ours = alg.pca_features(snips, 4)
    ref = SkPCA(n_components=4).fit(snips.reshape(200, -1))
    assert np.allclose(np.abs(ours.values), np.abs(ref.transform(snips.reshape(200, -1))), atol=1e-9)
    assert np.allclose(ours.explained, ref.explained_variance_ratio_, atol=1e-9) and ours.components.shape == (4, 90)
    assert np.allclose(ours.values.mean(axis=0), 0.0, atol=1e-9)


def test_feature_kinds_have_the_documented_shapes():
    snips = np.random.default_rng(4).standard_normal((50, 3, 30))
    assert alg.extract_features(snips, "raw").values.shape == (50, 90) and alg.extract_features(snips, "amplitude").values.shape == (50, 3)
    assert np.allclose(alg.extract_features(snips, "amplitude").values, snips.min(axis=2)) and alg.extract_features(snips, "pca", 2).values.shape == (50, 2)


# --- k-means und Silhouette -----------------------------------------------------------------------------------------------------------


def test_kmeans_recovers_well_separated_blobs_like_scikit_learn():
    F, truth, _ = _blobs()
    ours = alg.kmeans(F, 3, seed=1)
    ref = SkKMeans(3, n_init=10, random_state=0).fit(F)
    assert abs(ours.inertia - ref.inertia_) < 1e-6 * ref.inertia_
    conf = np.zeros((3, 3))
    for a, b in zip(truth, ours.labels):
        conf[a, b] += 1
    assert (conf.max(axis=1) == 80).all() and len(ours.restart_inertias) == C.N_RESTARTS


def test_kmeans_is_deterministic_orders_clusters_by_the_first_feature_and_handles_k_larger_than_n():
    F, _, _ = _blobs()
    a, b = alg.kmeans(F, 3, seed=2), alg.kmeans(F, 3, seed=2)
    assert np.array_equal(a.labels, b.labels) and (np.diff(a.centers[:, 0]) >= 0).all()
    assert alg.kmeans(F[:2], 5).centers.shape[0] == 2


def test_silhouette_matches_scikit_learn():
    F, truth, _ = _blobs(spread=0.9)
    assert abs(alg.silhouette(F, truth) - silhouette_score(F, truth)) < 1e-9
    km = alg.kmeans(F, 4, seed=1).labels
    assert abs(alg.silhouette(F, km) - silhouette_score(F, km)) < 1e-9 and alg.silhouette(F, np.zeros(len(F), dtype=int)) == 0.0


def test_silhouette_uses_a_fixed_subsample_for_large_inputs():
    F, truth, _ = _blobs(per=800)
    assert alg.silhouette(F, truth) == alg.silhouette(F, truth) and 0.0 < alg.silhouette(F, truth) <= 1.0


def test_choose_k_picks_the_true_number_for_clean_blobs_and_returns_the_curve():
    F, _, _ = _blobs(k=4, spread=0.1)
    k, curve = alg.choose_k(F, 7, seed=1)
    assert k == 4 and sorted(curve) == [2, 3, 4, 5, 6, 7] and curve[4] == max(curve.values())


