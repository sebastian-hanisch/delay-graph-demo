"""Auswertung der Verzögerungsgraph-Demo: für alle Verfahren (Verzögerungsgraph, Pipeline, Vorlagenabgleich) dieselbe Auswertung von Ereignissen (Zeit, Neuron) gegen die wahren Feuerzeiten,
Spitzen-F1 auch für ICA und SCA (Schätzspur -> Spitzen), die Fehlermaße der Dissertation für die Nachbarschaften, Sweeps, Szenen und das Urteil."""

import time
from dataclasses import dataclass

import numpy as np

import dg_algorithm as alg
import dg_constants as C
import dg_delay as dd
import dg_ica as ica
import dg_matching as tmm
import dg_sca as sca
import dg_scenario as sc


@dataclass(frozen=True)
class Settings:
    min_clique: int = 1                      # kleinste Cliquengröße des Verzögerungsgraphen (1 wie im heuristischen Ansatz der Dissertation, Abschn. 2.4: Annahme 2.3.3 wird dort nicht benutzt; 2 = nur Spikes mit bestätigter Kante)
    delay_assumed: int = -1                  # angenommene Schranke d_max der Verzögerungen (-1 = die wahre, wie in der Dissertation als bekannt vorausgesetzt)
    threshold: float = C.DEFAULT_MATCH_THRESHOLD
    contrast: str = C.DEFAULT_CONTRAST
    init_start: int = C.DEFAULT_INIT_START


def make_dataset(m=C.DEFAULT_N_CELLS, grid=C.DEFAULT_GRID, contact_p=C.DEFAULT_CONTACT_P, similarity=C.DEFAULT_SIMILARITY, delay_max=C.DEFAULT_DELAY_MAX, jitter=C.DEFAULT_JITTER,
                 rate_scale=C.DEFAULT_RATE_SCALE, noise=C.DEFAULT_NOISE, seconds=C.DEFAULT_SECONDS, seed=C.DEFAULT_SEED):
    return sc.make_dataset(m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed)


# --- Optimale Zuordnung (Ungarische Methode) ------------------------------------------------------------------------------------------


def assign(C_abs):
    """Optimale Zuordnung Zeile -> Spalte (jede Spalte höchstens einer Zeile), Summe der Einträge maximal (Ungarische Methode, O(n^3); die Bitmasken-DP der Vorgänger wächst mit 2^Spalten und ist bei 30 Zellen unbrauchbar).
    Rückgabe: Liste je Zeile mit dem Index der Spalte oder -1 (nicht zugeordnet: keine Spalte übrig oder Gewinn 0)."""
    A = np.asarray(C_abs, dtype=float)
    k, nc = A.shape
    N = max(k, nc)
    cost = np.zeros((N + 1, N + 1))
    cost[1: k + 1, 1: nc + 1] = -A
    u, v = np.zeros(N + 1), np.zeros(N + 1)
    p, way = np.zeros(N + 1, dtype=int), np.zeros(N + 1, dtype=int)
    for i in range(1, N + 1):
        p[0] = i
        j0 = 0
        minv = np.full(N + 1, np.inf)
        used = np.zeros(N + 1, bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, N + 1):
                if not used[j]:
                    cur = cost[i0, j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(N + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    out = [-1] * k
    for j in range(1, N + 1):
        i = p[j]
        if 1 <= i <= k and j <= nc and A[i - 1, j - 1] > 1e-12:
            out[i - 1] = j - 1
    return out


# --- Vergleich der Spurschätzer (ICA, SCA) -------------------------------------------------------------------------------------------------


def correlation_matrix(S, estimates):
    """|Korrelation| (k, nc) zwischen wahren Quellen und Schätzungen."""
    a = (S - S.mean(axis=1, keepdims=True)) / S.std(axis=1, keepdims=True)
    b = estimates - estimates.mean(axis=1, keepdims=True)
    sd = b.std(axis=1, keepdims=True)
    b = b / np.where(sd > 0, sd, 1.0)
    return np.abs(a @ b.T) / S.shape[1]


def matched(S, estimates):
    """Zuordnung + Vorzeichen-/Skalenkorrektur per Regression: (Zuordnung, |Korrelation| je Quelle, Schätzquellen in Skala und Vorzeichen der wahren Quellen)."""
    cm = correlation_matrix(S, estimates)
    idx = assign(cm)
    corr = np.array([cm[i, j] if j >= 0 else 0.0 for i, j in enumerate(idx)])
    aligned = np.zeros_like(S)
    for i, j in enumerate(idx):
        if j >= 0:
            e = estimates[j] - estimates[j].mean()
            aligned[i] = (e @ (S[i] - S[i].mean()) / (e @ e)) * e + S[i].mean()
    return idx, corr, aligned


def detect_spikes(x):
    """Negative Spitzen von x (Vorzeichen bereits wie bei der Zelle): tiefste zuerst, Mindestabstand; Schwelle max(4 sigma_MAD, 0.3 x typische Tiefe); typische Tiefe = Median der (höchstens) 10 tiefsten Spitzen."""
    sigma = np.median(np.abs(x - np.median(x))) / 0.6745
    threshold = -4.0 * sigma
    taken = np.zeros(len(x), bool)
    peaks = []
    for t in np.argsort(x):
        if x[t] > threshold:
            break
        if taken[max(0, t - C.DETECT_MIN_SEPARATION): t + C.DETECT_MIN_SEPARATION + 1].any():
            continue
        taken[t] = True
        peaks.append(t)
        if len(peaks) == 10:                                             # typische Tiefe = Median der 10 tiefsten Spitzen (ab hier gilt die Schwelle laufend)
            threshold = min(threshold, 0.3 * float(np.median(x[peaks])))
    if 0 < len(peaks) < 10:                                              # weniger als 10 Spitzen: typische Tiefe = Median der gefundenen
        threshold = min(threshold, 0.3 * float(np.median(x[peaks])))
        peaks = [t for t in peaks if x[t] <= threshold]
    return np.sort(np.array(peaks, dtype=int))


def spike_f1(detected, true, tolerance=C.DETECT_TOLERANCE):
    """F1 der erkannten gegen die wahren Spitzenzeiten (Zuordnung je wahrer Spitze zur nächsten, jede erkannte höchstens einmal)."""
    if len(detected) == 0 or len(true) == 0:
        return 0.0
    used = np.zeros(len(detected), bool)
    tp = 0
    for t in true:
        d = np.abs(detected - t)
        j = int(np.argmin(d))
        if d[j] <= tolerance and not used[j]:
            used[j] = True
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(detected), tp / len(true)
    return 2 * precision * recall / (precision + recall)


# --- Zuordnung erkannter und wahrer Spikes ---------------------------------------------------------------------------------------------


def truth_spikes(ds):
    """Alle wahren Spikes nach Zeit sortiert: (Zeit der ersten Spitze, Zelle, Kollision). Kollision = ein anderer Spike feuert höchstens COLLISION_WINDOW Abtastwerte vor oder nach diesem."""
    times = np.concatenate(ds.spike_times)
    cell = np.concatenate([np.full(len(t), i) for i, t in enumerate(ds.spike_times)])
    fire = np.concatenate(ds.firing_times)
    order = np.argsort(times, kind="stable")
    times, cell, fire = times[order], cell[order], fire[order]
    all_fire = np.sort(fire)
    left = np.searchsorted(all_fire, fire - C.COLLISION_WINDOW, side="left")
    right = np.searchsorted(all_fire, fire + C.COLLISION_WINDOW, side="right")
    return times, cell, (right - left) > 1


def match_detections(detected, truth_times, tolerance=C.MATCH_TOLERANCE):
    """Je erkanntem Ereignis der Index der nächsten noch freien wahren Spitze innerhalb `tolerance`, sonst -1 (jede wahre Spitze höchstens einmal)."""
    out = np.full(len(detected), -1, dtype=int)
    used = np.zeros(len(truth_times), bool)
    for i, t in enumerate(detected):
        j = int(np.searchsorted(truth_times, t))
        best, best_d = -1, tolerance + 1
        for c in (j - 2, j - 1, j, j + 1):
            if 0 <= c < len(truth_times) and not used[c] and abs(int(truth_times[c]) - int(t)) < best_d:
                best, best_d = c, abs(int(truth_times[c]) - int(t))
        if best >= 0 and best_d <= tolerance:
            out[i] = best
            used[best] = True
    return out


@dataclass(frozen=True)
class EventResult:
    """Sortiergüte einer Liste von Ereignissen (Zeit, Cluster/Vorlage/Charakteristik); für alle Verfahren dieselbe Auswertung."""
    recall: float                 # Anteil der wahren Spikes, die erkannt wurden
    precision: float              # Anteil der Ereignisse, die zu einem wahren Spike gehören
    n_detected: int
    n_true: int
    n_ghosts: int
    accuracy: float               # richtig sortierter Anteil der erkannten wahren Spikes
    accuracy_single: float
    accuracy_collision: float
    confusion: np.ndarray         # (m, k)
    cluster_of_neuron: list       # je Zelle das zugeordnete Cluster (-1 = keins)
    neuron_of_event: np.ndarray
    truth_of_event: np.ndarray
    collision_of_event: np.ndarray
    f1: float                     # mittlerer Spitzen-F1 der Zellen (Ereignisse ihres Clusters gegen die wahren Zeiten)
    f1_per_neuron: np.ndarray
    collision_share: float
    majority_baseline: float


def evaluate_events(ds, times, labels, k):
    m = ds.n_neurons
    times = np.asarray(times, dtype=int)
    labels = np.asarray(labels, dtype=int)
    k = max(int(k), 1)
    tt, tn, tc = truth_spikes(ds)
    mi = match_detections(times, tt)
    ok = mi >= 0
    matched_truth = np.full(len(times), -1)
    matched_truth[ok] = tn[mi[ok]]
    collision = np.zeros(len(times), bool)
    collision[ok] = tc[mi[ok]]
    confusion = np.zeros((m, k))
    for lab, y in zip(labels[ok], tn[mi[ok]]):
        confusion[y, lab] += 1
    idx = assign(confusion) if ok.any() else [-1] * m
    cmap = {c: i for i, c in enumerate(idx) if c >= 0}
    neuron_of_event = np.array([cmap.get(int(l), -1) for l in labels], dtype=int) if len(labels) else np.zeros(0, dtype=int)
    correct = (neuron_of_event == matched_truth) & ok
    n_ok = max(int(ok.sum()), 1)
    single, coll = ok & ~collision, ok & collision
    f1s = []
    for i in range(m):
        c = idx[i]
        detected_i = times[labels == c] if c >= 0 else np.array([], dtype=int)
        f1s.append(spike_f1(detected_i, ds.spike_times[i]))
    counts = np.bincount(tn, minlength=m)
    return EventResult(
        recall=float(ok.sum() / max(len(tt), 1)), precision=float(ok.sum() / max(len(times), 1)), n_detected=int(len(times)), n_true=int(len(tt)), n_ghosts=int((~ok).sum()),
        accuracy=float(correct.sum() / n_ok), accuracy_single=float(correct[single].sum() / max(single.sum(), 1)) if single.any() else float("nan"),
        accuracy_collision=float(correct[coll].sum() / max(coll.sum(), 1)) if coll.any() else float("nan"),
        confusion=confusion, cluster_of_neuron=list(idx), neuron_of_event=neuron_of_event, truth_of_event=matched_truth, collision_of_event=collision,
        f1=float(np.mean(f1s)), f1_per_neuron=np.array(f1s), collision_share=float(tc.mean()), majority_baseline=float(counts.max() / max(counts.sum(), 1)))


# --- Eigenschaften der Daten, die die Methode begrenzen (Dissertation: Prop. 2.3.1, Annahme 2.3.3) --------------------------------------------------


def single_contact_share(ds):
    """Anteil der Zellen mit nur einer Kontaktelektrode (Annahme 2.3.3 verletzt: ihre Spikes lassen sich über Verzögerungen nicht erkennen)."""
    return float(np.mean([len(n) == 1 for n in ds.neighbourhoods]))


def indistinguishable_pairs(ds):
    """Zahl der Zellenpaare mit identischer normierter Charakteristik (Toleranz 0): im Modell der Dissertation prinzipiell nicht trennbar (Prop. 2.3.1)."""
    chars = [c for c in ds.characteristics]
    return sum(chars[a] == chars[b] for a in range(len(chars)) for b in range(a + 1, len(chars)))


def characteristic_errors(ds, sorting):
    """Relativer Fehler der Charakteristiken (zeta, Gl. 2.12; Übereinstimmung mit Toleranz 2 epsilon): fehlende plus falsche Charakteristiken, geteilt durch die Zahl der wahren (verschiedenen)."""
    true = {}
    for ch in ds.characteristics:
        if len(ch) >= 2:
            true[tuple(j for j, _ in ch), tuple(r for _, r in ch[1:])] = True
    est = [(e, tuple(int(x) for x in v)) for e, v, _ in sorting.candidates]

    def near(a, b):
        return a[0] == b[0] and all(abs(x - y) <= dd.TOLERANCE for x, y in zip(a[1], b[1]))

    missing = sum(not any(near(t, e) for e in est) for t in true)
    false = sum(not any(near(t, e) for t in true) for e in est)
    return (missing + false) / max(len(true), 1)


# --- Gesamtanalyse ------------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Analysis:
    ds: sc.Dataset
    settings: Settings
    delay: dd.DelayResult
    dg: EventResult               # Verzögerungsgraph
    eta_m: float                  # fehlende Nachbarschaften (relativ)
    eta_f: float                  # falsche Nachbarschaften (relativ)
    zeta: float                   # relativer Fehler der Charakteristiken
    pipe: EventResult             # Standardpipeline
    matching: object              # dg_matching.Matching
    tm: EventResult               # Vorlagenabgleich
    comparators: dict             # "ica", "sca" -> {"f1"}
    seconds: dict                 # Rechenzeit je Verfahren
    single_contact: float
    ambiguous_pairs: int


def pipeline_components(ds):
    """Zahl der PCA-Komponenten der Pipeline: die Zellzahl (mindestens 3, höchstens PCA_COMPONENTS_MAX) - drei wie in den Vorgängern reichen für ein Dutzend Zellen nicht."""
    return int(min(max(ds.n_neurons, 3), C.PCA_COMPONENTS_MAX))


def n_components(ds):
    return min(ds.n_electrodes, ds.n_neurons)


def comparator_f1(ds, settings, names=("ica", "sca")):
    """Mittlerer Spitzen-F1 der Zellen für ICA und - nur wenn es mehr Zellen als Elektroden gibt, dem Anwendungsbereich der SCA - für SCA (Schätzung -> Zuordnung per Korrelation -> Schwelle -> Treffer) und ihre Rechenzeit.
    Gemessen: bei mehr Elektroden als Zellen ist die SCA mit den Einstellungen der sca-demo (für 2-8 Elektroden abgestimmt) unbrauchbar (Aktivitätsschwelle 4 x Rausch-Norm, die mit der Wurzel der Elektrodenzahl wächst) und mit 12 s je Analyse zu langsam."""
    m, nc = ds.n_neurons, n_components(ds)
    out, secs, estimates = {}, {}, {}
    t0 = time.perf_counter()
    if "ica" in names:
        estimates["ica"] = ica.fit_ica(ds.X, nc, settings.contrast, C.DEFAULT_METHOD, settings.init_start).sources
        secs["ica"] = time.perf_counter() - t0
    if "sca" in names and m > ds.n_electrodes:
        t0 = time.perf_counter()
        estimates["sca"] = sca.fit_sca(ds.X, m, "l1", seed=settings.init_start).sources
        secs["sca"] = time.perf_counter() - t0
    for name, est in estimates.items():
        idx, corr, aligned = matched(ds.S, est)
        out[name] = {"f1": float(np.mean([spike_f1(detect_spikes(aligned[i]), ds.spike_times[i]) for i in range(m)])), "corr": float(corr.mean())}
    return out, secs


def analyse(ds, settings, with_comparators=True, with_waveform=True, comparators_used=("ica", "sca")):
    """Alle Verfahren auf einem Datensatz. `with_waveform=False` lässt Pipeline und Abgleich aus (dann sind pipe/tm/matching None)."""
    secs = {}
    t0 = time.perf_counter()
    bound = ds_delay_bound(ds, settings)
    delay = dd.run_delay_graph(ds.X, ds.positions, ds.nu, bound, settings.min_clique)
    secs["dg"] = time.perf_counter() - t0
    dg = evaluate_events(ds, delay.sorting.times, delay.sorting.labels, max(len(delay.sorting.candidates), 1))
    eta_m, eta_f = dd.neighbourhood_errors(delay.neighbourhoods, sc.neighbourhood_true(ds))
    zeta = characteristic_errors(ds, delay.sorting)
    pipe = tm = matching = None
    if with_waveform:
        t0 = time.perf_counter()
        sorting = alg.sort_spikes(ds.X, ds.n_neurons, n_components=pipeline_components(ds), cluster_mode="known", seed=settings.init_start)
        secs["pipe"] = time.perf_counter() - t0
        pipe = evaluate_events(ds, sorting.times, sorting.clustering.labels, sorting.k)
        t0 = time.perf_counter()
        matching = tmm.run_matching(ds.X, ds.n_neurons, settings.threshold, C.DEFAULT_MIN_AMPLITUDE, C.DEFAULT_REFINE, C.DEFAULT_ROUNDS, "known", settings.init_start, sorting)
        secs["matching"] = time.perf_counter() - t0 + secs["pipe"]
        tm = evaluate_events(ds, matching.pursuit.times, matching.pursuit.templates, max(matching.templates.shape[0], 1))
    comparators = {}
    if with_comparators:
        comparators, cs = comparator_f1(ds, settings, comparators_used)
        secs.update(cs)
    return Analysis(ds, settings, delay, dg, eta_m, eta_f, zeta, pipe, matching, tm, comparators, secs, single_contact_share(ds), indistinguishable_pairs(ds))


def ds_delay_bound(ds, settings):
    """Angenommene Schranke d_max der Verzögerungen: die eingestellte Verzögerungsspanne (die Dissertation setzt die Schranke als bekannt voraus), sofern nicht überschrieben."""
    return settings.delay_assumed if settings.delay_assumed >= 0 else ds.delay_max


def analyse_for(params, settings, with_comparators=True, with_waveform=True, comparators_used=("ica", "sca")):
    """`params` = (m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed)."""
    return analyse(make_dataset(*params), settings, with_comparators, with_waveform, comparators_used)


# --- Sweeps und Szenen -----------------------------------------------------------------------------------------------------------------------

SWEEP_VALUES = {
    "n_cells": (6, 12, 20, 30),
    "grid": (3, 4, 5, 6),
    "contact_p": (0.1, 0.2, 0.35, 0.5),
    "similarity": (0.0, 0.5, 1.0),
    "delay_max": (0, 2, 4, 8),
    "jitter": (0.0, 1.0, 2.0, 3.0),
    "rate_scale": (0.25, 0.5, 1.0, 2.0, 4.0),
    "noise": (2.0, 10.0, 20.0, 30.0, 40.0),
    "seconds": (1.0, 2.0, 3.0, 6.0),
}
SWEEP_LABELS = {"n_cells": "Anzahl Zellen", "grid": "Gitterkantenlänge (Elektroden = Kante²)", "contact_p": "Kontaktwahrscheinlichkeit p", "similarity": "Wellenform-Ähnlichkeit (0 = alle gleich)",
                "delay_max": "Verzögerungsspanne [Abtastwerte]", "jitter": "Verzögerungs-Jitter [Abtastwerte]", "rate_scale": "Feuerraten-Faktor", "noise": "Rauschen [µV]", "seconds": "Länge der Aufnahme [s]"}
_DATA_KEYWORD = {"n_cells": "m", "grid": "grid", "contact_p": "contact_p", "similarity": "similarity", "delay_max": "delay_max", "jitter": "jitter", "rate_scale": "rate_scale", "noise": "noise", "seconds": "seconds"}
METHODS = ("dg", "pipe", "tm", "ica")


def _record(a):
    out = {"dg": a.dg.f1, "pipe": a.pipe.f1, "tm": a.tm.f1, "ica": a.comparators["ica"]["f1"] if "ica" in a.comparators else float("nan"),
           "recall": a.dg.recall, "precision": a.dg.precision, "accuracy": a.dg.accuracy, "pipe_accuracy": a.pipe.accuracy, "tm_accuracy": a.tm.accuracy,
           "eta_m": a.eta_m, "eta_f": a.eta_f, "zeta": a.zeta, "single_contact": a.single_contact, "ambiguous_pairs": float(a.ambiguous_pairs)}
    for k, v in a.seconds.items():
        out["sec_" + k] = v
    return out


def _summarise(x, per_seed):
    row = {"x": x}
    for key in per_seed[0]:
        arr = np.array([r.get(key, np.nan) for r in per_seed], dtype=float)
        ok = not np.isnan(arr).all()
        row[key] = float(np.nanmean(arr)) if ok else float("nan")
        row[key + "_std"] = float(np.nanstd(arr)) if ok else float("nan")
        row[key + "_min"] = float(np.nanmin(arr)) if ok else float("nan")
        row[key + "_max"] = float(np.nanmax(arr)) if ok else float("nan")
    return row


def sweep(parameter, values=None, settings=Settings(), **base):
    """Mittel, Streuung und Spanne (über die festen Sweep-Datensätze) der Kennzahlen aller Verfahren in Abhängigkeit von einem Regler (SCA nicht: nur im Anwendungsbereich m > n und zu langsam für Sweeps)."""
    values = SWEEP_VALUES[parameter] if values is None else values
    rows = []
    for x in values:
        per_seed = []
        for seed in C.SWEEP_SEEDS:
            kw = dict(base)
            kw[_DATA_KEYWORD[parameter]] = x
            per_seed.append(_record(analyse(make_dataset(seed=seed, **kw), settings, comparators_used=("ica",))))
        rows.append(_summarise(x, per_seed))
    return rows


SCENES = (
    ("Standardfall (Dissertations-Modell: 12 Zellen, 5×5 Elektroden)", dict()),
    ("Viele Kontakte, gleiche Formen (p 0.5, Ähnlichkeit 0, 20 Zellen, 4×4)", dict(contact_p=0.5, similarity=0.0, m=20, grid=4)),
    ("Mehr Zellen als Elektroden (30 Zellen, 5×5)", dict(m=30)),
    ("Hohe Feuerrate (Faktor 4, viele Überlappungen)", dict(rate_scale=4.0)),
    ("Starkes Rauschen (30 µV)", dict(noise=30.0)),
    ("Wackelnde Verzögerungen (Jitter 2)", dict(jitter=2.0)),
    ("Niedrige Feuerraten (Faktor 0.25, unter der angenommenen Untergrenze)", dict(rate_scale=0.25)),
    ("Viele Einzelkontakt-Zellen auf wenigen Elektroden (20 Zellen, 3×3, p 0.1)", dict(m=20, grid=3, contact_p=0.1)),
)
SCENE_METHODS = ("dg", "pipe", "tm", "ica")


def scene_table(settings=Settings(), **base):
    """Verzögerungsgraph, Pipeline, Vorlagenabgleich und ICA in acht Szenen (Spitzen-F1 der Zellen), Mittel und Spanne über die Sweep-Datensätze, dazu Rechenzeit des Verzögerungsgraphen und des Abgleichs."""
    rows = []
    for label, scene in SCENES:
        kw = dict(base)
        kw.update(scene)
        acc = {name: [] for name in SCENE_METHODS}
        acc.update({"single_contact": [], "ambiguous_pairs": [], "sec_dg": [], "sec_matching": []})
        for seed in C.SWEEP_SEEDS:
            a = analyse(make_dataset(seed=seed, **kw), settings, comparators_used=("ica",))
            for name, val in (("dg", a.dg.f1), ("pipe", a.pipe.f1), ("tm", a.tm.f1), ("ica", a.comparators["ica"]["f1"]), ("single_contact", a.single_contact), ("ambiguous_pairs", float(a.ambiguous_pairs)),
                              ("sec_dg", a.seconds["dg"]), ("sec_matching", a.seconds["matching"])):
                acc[name].append(val)
        row = {"scene": label}
        for name, vals in acc.items():
            row[name] = float(np.mean(vals))
            row[name + "_min"], row[name + "_max"] = float(np.min(vals)), float(np.max(vals))
        rows.append(row)
    return rows


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------

VERDICT_MARGIN = 0.08             # F1-Abstand zum besten anderen Verfahren, ab dem ein Sieger benannt wird
VERDICT_WEAK = 0.8                # F1 des Verzögerungsgraphen darunter: Ursache suchen (Referenzläufe)
VERDICT_DROP = 0.10               # Verlust gegenüber dem Referenzlauf, ab dem eine Ursache benannt wird
VERDICT_SINGLES = 0.2             # Anteil Einzelkontakt-Zellen, ab dem ihr Ausschluss als Ursache einer schwachen Leistung gilt


def _dg_f1(params, settings):
    ds = make_dataset(*params)
    r = dd.run_delay_graph(ds.X, ds.positions, ds.nu, ds_delay_bound(ds, settings), settings.min_clique)
    return evaluate_events(ds, r.sorting.times, r.sorting.labels, max(len(r.sorting.candidates), 1)).f1


def verdict(a):
    """(Art, Code, Kennzahlen). Vergleich nur mit großer Marge; ist der Verzögerungsgraph schwach, benennt ein Referenzlauf (nur dieses Verfahren, ohne die gestörte Eigenschaft) die Ursache:
    Jitter, zu niedrige Feuerrate, Rauschen, oder nicht unterscheidbare Zellen (identische Charakteristik, Prop. 2.3.1)."""
    ds = a.ds
    m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed = ds.params
    others = {"pipe": a.pipe.f1, "tm": a.tm.f1, **{k: v["f1"] for k, v in a.comparators.items()}}
    best_name = max(others, key=others.get)
    data = {"dg": a.dg.f1, "best": others[best_name], "best_name": best_name, **{f"{k}_f1": v for k, v in others.items()}, "recall": a.dg.recall, "precision": a.dg.precision, "accuracy": a.dg.accuracy,
            "eta_m": a.eta_m, "eta_f": a.eta_f, "single_contact": a.single_contact, "ambiguous_pairs": a.ambiguous_pairs, "n": ds.n_electrodes, "m": m, "jitter": jitter, "rate_scale": rate_scale, "noise": noise}
    if a.dg.f1 < VERDICT_WEAK:
        if a.settings.min_clique >= 2 and a.single_contact >= VERDICT_SINGLES:
            return "warning", "singles_excluded", data
        base = list(ds.params)
        refs = {}
        if jitter > 0:
            refs["jitter"] = _dg_f1(tuple(base[:5] + [0.0] + base[6:]), a.settings) - a.dg.f1
        if rate_scale < 1:
            refs["rate"] = _dg_f1(tuple(base[:6] + [1.0] + base[7:]), a.settings) - a.dg.f1
        if noise > C.DEFAULT_NOISE:
            refs["noise"] = _dg_f1(tuple(base[:7] + [C.DEFAULT_NOISE] + base[8:]), a.settings) - a.dg.f1
        if refs:
            cause, drop = max(refs.items(), key=lambda kv: kv[1])
            data["drop"] = drop
            if drop > VERDICT_DROP:
                return "warning", cause, data
        if a.ambiguous_pairs > 0:
            return "warning", "ambiguous", data
    if a.dg.f1 - others[best_name] >= VERDICT_MARGIN:
        return "success", "delay_wins", data
    if others[best_name] - a.dg.f1 >= VERDICT_MARGIN:
        return "warning", "waveform_wins", data
    return "success", "comparable", data
