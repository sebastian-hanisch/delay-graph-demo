"""Verzögerungsgraph (Dissertation, Kap. 2), numpy von Grund auf: Spitzenschätzung je Elektrode (Abschn. 2.2.2), geschätzter Verzögerungsgraph und Nachbarschaften (Abschn. 2.2.3, Algorithmen 1 und 2),
Greedy-Cliquen und Charakteristiken (Abschn. 2.3/2.4, Algorithmen 3 und 5). Nutzt keine Wellenform - nur die Zeitpunkte der Spitzen und die über die Elektroden stabilen Zeitdifferenzen.

Eigener Nachbau dieser Demo. Abweichungen von der Dissertation: (1) mit mehr Rauschen als in der Dissertation (10 muV) werden die Schwellen der Spitzenschätzung entsprechend angehoben, bei weniger bleiben sie wie dort,
(2) die Charakteristiken werden mit Toleranz 2*epsilon gruppiert (die Dissertation behandelt das Spike-Sorting nur für exakte Spitzenzeiten), (3) Cliquen, die in einer größeren enthalten sind, zählen nicht als eigener Spike.
Wie in der Heuristik der Dissertation (Abschn. 2.4) zählen auch Cliquen aus einem einzigen Knoten (Zelle mit einer Elektrode), solange `min_clique` = 1 ist."""

from dataclasses import dataclass
from math import comb

import numpy as np

import dg_constants as C

TOLERANCE = 2 * C.EPSILON                     # Toleranz der Charakteristiken (Abweichung je Komponente)
PAIR_DISTANCE = 2 * C.RADIUS                  # Elektroden weiter als 2R haben keine gemeinsame Zelle


def noise_sigma(x):
    """Robuste Rausch-Standardabweichung (MAD / 0.6745); Untergrenze 2 muV, damit rauschfreie Daten keine Schwelle 0 ergeben."""
    return max(float(np.median(np.abs(x - np.median(x))) / 0.6745), 2.0)


def _binomial_kernel(b):
    k = np.array([comb(2 * b, i) for i in range(2 * b + 1)], dtype=float)
    return k / k.sum()


def peak_signals(x, sigma=None):
    """Die Zwischenergebnisse der Spitzenschätzung: geglättete Werte, geglättete Differenzfolge und die beiden Schwellen (wie in der Dissertation in muV; bei mehr Rauschen als 10 muV mit dem geschätzten Rauschen angehoben)."""
    sigma = noise_sigma(x) if sigma is None else sigma
    scale = max(1.0, sigma / C.NOISE_REFERENCE_UV)                    # bei weniger Rauschen als in der Dissertation bleiben die absoluten Schwellen (sonst würde der 5-%-Hintergrund als Spike erkannt)
    d, u = C.DIFF_D, max(1, C.DIFF_D // 6)
    smooth = np.convolve(x, _binomial_kernel(C.FILTER_B), mode="same")
    diff = smooth.copy()
    diff[d:] = smooth[d:] - smooth[:-d]
    diff = np.convolve(diff, np.ones(2 * u + 1) / (2 * u + 1), mode="same")
    return smooth, diff, C.ETA_MINUS_UV * scale, C.ETA_PLUS_UV * scale


def estimate_peaks(x, sigma=None):
    """Spitzenschätzung einer Elektrode (Abschn. 2.2.2): Binomialfilter, Differenz r_t - r_(t-d), Rechteckfilter, Schwellen eta- / eta+; Spitze = Minimum der geglätteten Werte zwischen dem steilen Abstieg und dem folgenden Anstieg.
    Rückgabe: sortierte Spitzenzeiten."""
    T = len(x)
    smooth, diff, eta_minus, eta_plus = peak_signals(x, sigma)
    d = C.DIFF_D
    candidates = np.flatnonzero(diff < eta_minus)
    peaks = []
    i = 0
    while i < len(candidates):
        t = int(candidates[i])
        hi = min(t + 2 * d, T - 1)
        window = diff[t + d // 2: hi + 1]
        above = np.flatnonzero(window > eta_plus)
        if len(above):
            end = t + d // 2 + int(above[0])
            peaks.append(t + int(np.argmin(smooth[t: end + 1])))
            i = int(np.searchsorted(candidates, end + 1))
        else:
            i += 1
    return np.array(sorted(set(peaks)), dtype=int)


def electrode_distances(positions):
    diff = positions[:, None, :] - positions[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


@dataclass(frozen=True)
class Graph:
    peaks: tuple                  # je Elektrode die geschätzten Spitzenzeiten (Knoten (j, l))
    adjacency: dict               # (j, l) -> {j2: set(l2)} Kanten des geschätzten Verzögerungsgraphen
    pair_differences: dict        # (j1, j2) -> (Differenzen aller Paare in +-delta, zulässige Maske)
    n_edges: int
    delta: int
    nu: int


def bounded_differences(p1, p2, delta):
    """Algorithmus 1: alle Differenzen p2 - p1 mit Betrag <= delta. Rückgabe (l1, l2, Differenz)."""
    if len(p1) == 0 or len(p2) == 0:
        z = np.zeros(0, dtype=int)
        return z, z, z
    lo = np.searchsorted(p1, p2 - delta, side="left")
    hi = np.searchsorted(p1, p2 + delta, side="right")
    counts = hi - lo
    l2 = np.repeat(np.arange(len(p2)), counts)
    l1 = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if counts.sum() else np.zeros(0, dtype=int)
    return l1, l2, p2[l2] - p1[l1]


def admissible_mask(diffs, nu, window=4 * C.EPSILON, theta1=C.THETA1):
    """Algorithmus 2: eine Differenz ist zulässig, wenn sie in einem Intervall der Länge 4*epsilon liegt, das insgesamt mindestens (1 - theta1) * nu Differenzen enthält (Vielfachheiten mitgezählt)."""
    if len(diffs) == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(diffs, kind="stable")
    d = diffs[order]
    end = np.searchsorted(d, d + window, side="right")
    starts = np.flatnonzero(end - np.arange(len(d)) >= (1 - theta1) * nu)
    cover = np.zeros(len(d) + 1, dtype=int)
    np.add.at(cover, starts, 1)
    np.add.at(cover, end[starts], -1)
    mask_sorted = np.cumsum(cover)[:-1] > 0
    mask = np.zeros(len(d), dtype=bool)
    mask[order] = mask_sorted
    return mask


def build_graph(peaks, positions, nu, delay_max):
    """Geschätzter Verzögerungsgraph: Kante zwischen Spitzen zweier Elektroden (Abstand <= 2R), wenn ihre Zeitdifferenz zulässig ist. delta = d_max + 2*epsilon (Gleichung 2.10)."""
    n = len(peaks)
    delta = max(delay_max, 0) + 2 * C.EPSILON
    dist = electrode_distances(positions)
    adjacency = {(j, l): {} for j in range(n) for l in range(len(peaks[j]))}
    pair_diffs, n_edges = {}, 0
    for j1 in range(n):
        for j2 in range(j1 + 1, n):
            if dist[j1, j2] > PAIR_DISTANCE or len(peaks[j1]) == 0 or len(peaks[j2]) == 0:
                continue
            l1, l2, diffs = bounded_differences(peaks[j1], peaks[j2], delta)
            mask = admissible_mask(diffs, nu)
            pair_diffs[(j1, j2)] = (diffs, mask)
            for a, b in zip(l1[mask], l2[mask]):
                adjacency[(j1, int(a))].setdefault(j2, set()).add(int(b))
                adjacency[(j2, int(b))].setdefault(j1, set()).add(int(a))
                n_edges += 1
    return Graph(tuple(peaks), adjacency, pair_diffs, n_edges, delta, nu)


def node_neighbourhood(graph, node):
    """N~(j, l): die Elektrode des Knotens und alle Elektroden, mit denen er eine Kante hat."""
    return frozenset({node[0]} | {j2 for j2, ls in graph.adjacency[node].items() if ls})


def estimate_neighbourhoods(graph, theta2=C.THETA2):
    """Geschätzte Nachbarschaftsmenge: Mengen, die mindestens (1 - theta2) * nu * |N| Mal als Nachbarschaft eines Knotens vorkommen. Rückgabe (Menge -> Häufigkeit, nur die bestätigten)."""
    counts = {}
    for node in graph.adjacency:
        nb = node_neighbourhood(graph, node)
        counts[nb] = counts.get(nb, 0) + 1
    return {nb: c for nb, c in counts.items() if c >= (1 - theta2) * graph.nu * len(nb)}, counts


def neighbourhood_errors(estimated, true_sets):
    """Relative Zahl fehlender (eta_m) und falscher (eta_f) Nachbarschaften, wie in der Dissertation."""
    est = set(estimated)
    return len(true_sets - est) / max(len(true_sets), 1), len(est - true_sets) / max(len(true_sets), 1)


def greedy_clique(graph, node):
    """Algorithmus 5: aus dem Nachbarschaftsgraphen des Knotens iterativ den Knoten mit dem größten Grad hinzunehmen und auf dessen Nachbarn einschränken."""
    adj = graph.adjacency
    clique = [node]
    remaining = [(j2, l2) for j2, ls in adj[node].items() for l2 in ls]

    def linked(a, b):
        return b[1] in adj[a].get(b[0], ())

    while remaining:
        degrees = [sum(linked(v, w) for w in remaining if w != v) for v in remaining]
        best = remaining[int(np.argmax(degrees))]
        clique.append(best)
        remaining = [w for w in remaining if w != best and linked(best, w)]
    return clique


def characteristic(graph, clique):
    """Normierte Charakteristik einer Clique: (Elektroden sortiert, Zeitdifferenzen zur Elektrode mit kleinstem Index)."""
    nodes = sorted(clique)
    ref = graph.peaks[nodes[0][0]][nodes[0][1]]
    return tuple(j for j, _ in nodes), np.array([graph.peaks[j][l] - ref for j, l in nodes[1:]], dtype=int), nodes


def group_characteristics(items, nu, tolerance=TOLERANCE, fraction=C.CANDIDATE_FRACTION):
    """Kandidaten unter den Charakteristiken einer Elektrodenmenge: Modi der Vektoren (Abweichung <= tolerance je Komponente), die mindestens fraction * nu Mal vorkommen. Rückgabe Liste (Vektor, Mitglieder-Indizes)."""
    V = np.array([v for v, _ in items], dtype=int).reshape(len(items), -1)
    left = np.ones(len(items), bool)
    out = []
    while left.any():
        idx = np.flatnonzero(left)
        if V.shape[1] == 0:
            near = np.ones((len(idx), len(idx)), bool)
        else:
            near = (np.abs(V[idx][:, None, :] - V[idx][None, :, :]).max(axis=2)) <= tolerance
        counts = near.sum(axis=1)
        best = int(np.argmax(counts))
        if counts[best] < fraction * nu:
            break
        members = idx[near[best]]
        out.append((np.rint(np.median(V[members], axis=0)).astype(int), [items[m][1] for m in members]))
        left[members] = False
    return out


@dataclass(frozen=True)
class Sorting:
    times: np.ndarray             # Zeit je Ereignis (Spitze auf der Elektrode mit kleinstem Index der Clique)
    labels: np.ndarray            # Nummer der Charakteristik (Neuron) je Ereignis
    cliques: tuple                # je Ereignis die Knoten der Clique
    candidates: tuple             # je Neuron (Elektroden, Vektor der relativen Verzögerungen, Häufigkeit)
    n_cliques: int                # verschiedene maximale Cliquen
    n_singletons: int             # davon mit nur einem Knoten
    n_unassigned: int             # Cliquen ohne bestätigte Charakteristik (nicht gemeldet)


def sort_spikes(graph, min_clique=1, tolerance=TOLERANCE):
    """Spike-Sorting (Abschn. 2.3/2.4 mit Greedy-Clique): je Knoten die Greedy-Clique, nur maximale und verschiedene behalten, Charakteristiken gruppieren, Kandidaten (>= nu/2 Mal) sind die Neuronen,
    jede zugeordnete Clique ist ein Spike des Neurons. `min_clique` = kleinste Cliquengröße (1 lässt Spitzen ohne bestätigte Kante als Neuronen mit einer Elektrode zu)."""
    seen = {}
    for node in graph.adjacency:
        clique = greedy_clique(graph, node)
        seen[frozenset(clique)] = clique
    sets = sorted(seen, key=len, reverse=True)
    kept = []
    for s in sets:
        if not any(s < big for big in kept):
            kept.append(s)
    n_singletons = sum(len(s) == 1 for s in kept)
    usable = [s for s in kept if len(s) >= min_clique]
    groups = {}
    for s in usable:
        electrodes, vec, nodes = characteristic(graph, s)
        groups.setdefault(electrodes, []).append((vec, (s, nodes)))
    times, labels, cliques, candidates = [], [], [], []
    assigned = 0
    for electrodes in sorted(groups):
        for vec, members in group_characteristics(groups[electrodes], graph.nu, tolerance):
            k = len(candidates)
            candidates.append((electrodes, vec, len(members)))
            for s, nodes in members:
                times.append(int(graph.peaks[nodes[0][0]][nodes[0][1]]))
                labels.append(k)
                cliques.append(tuple(nodes))
                assigned += 1
    order = np.argsort(times, kind="stable")
    return Sorting(np.array(times, dtype=int)[order], np.array(labels, dtype=int)[order], tuple(cliques[i] for i in order), tuple(candidates), len(kept), n_singletons, len(usable) - assigned)


@dataclass(frozen=True)
class DelayResult:
    graph: Graph
    neighbourhoods: dict          # bestätigte Nachbarschaften -> Häufigkeit
    all_neighbourhoods: dict      # alle vorkommenden Mengen -> Häufigkeit
    sorting: Sorting
    sigma: float


def run_delay_graph(X, positions, nu, delay_max, min_clique=1):
    """Gesamtverfahren: Spitzen je Elektrode -> Verzögerungsgraph -> Nachbarschaften -> Cliquen -> Neuronen. `delay_max` ist die (angenommene) obere Schranke der Verzögerungen."""
    sigma = float(np.median([noise_sigma(x) for x in X]))
    peaks = [estimate_peaks(x, sigma) for x in X]
    graph = build_graph(peaks, positions, nu, delay_max)
    confirmed, allnb = estimate_neighbourhoods(graph)
    return DelayResult(graph, confirmed, allnb, sort_spikes(graph, min_clique), sigma)
