"""Szenario der Verzögerungsgraph-Demo nach dem Modell der Dissertation (Abschn. 2.1): Elektrodengitter, Zellen mit zufälligen Kontakten und Verzögerungen, Zwei-Gauß-Vorlage je Kontakt, Erneuerungsprozess als Feuern,
weißes Rauschen plus Hintergrund pseudo-adjazenter Zellen. Neu gegenüber der Dissertation (als Regler, Standard = wie dort): Wellenform-Ähnlichkeit, Verzögerungs-Jitter, Feuerraten-Faktor.

Nur numpy. Jede Zelle hat eigene Zufallsströme (Seed, Nummer): Zelle 0 sieht für einen Seed immer gleich aus, egal wie viele Zellen eingestellt sind."""

from dataclasses import dataclass

import numpy as np

import dg_constants as C


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray                 # (n, T) Elektrodensignale in muV (mit Rauschen)
    X_clean: np.ndarray           # (n, T) ohne Rauschen (mit Hintergrund)
    S: np.ndarray                 # (m, T) je Zelle eine Spur (Vorlage der stärksten Kontaktelektrode an den Spitzenzeiten), Varianz 1 - nur für den Vergleich mit ICA und SCA
    n_neurons: int                # m
    n_electrodes: int             # n
    grid: int
    seed: int
    noise_sigma: float
    params: tuple                 # (m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed): womit der Datensatz erzeugt wurde
    delay_max: int                # obere Schranke der Verzögerungen (Regler)
    jitter: float
    positions: np.ndarray         # (n, 2) Elektrodenzentren
    cell_positions: np.ndarray    # (m, 2)
    neighbourhoods: tuple         # je Zelle: sortiertes Array der Kontaktelektroden N_i
    delays: tuple                 # je Zelle: Verzögerungen d_ij zu diesen Elektroden
    firing_times: tuple           # je Zelle: Feuerzeiten s_ik
    spike_times: tuple            # je Zelle: Zeit der ersten Spitze s_ik + min_j d_ij (die Zeit, an der ein Verfahren den Spike melden soll)
    peak_times: tuple             # je Zelle: (p_i, |N_i|) tatsächliche Spitzenzeiten s_ik + d_ij (+ Jitter) an ihren Kontaktelektroden
    electrode_peaks: tuple        # je Elektrode: sortierte Spitzenzeiten aller Zellen (wahre Knoten des Verzögerungsgraphen)
    electrode_peak_cell: tuple    # je Elektrode: zugehörige Zelle je Spitze
    electrode_peak_spike: tuple   # je Elektrode: zugehöriger Spike-Index der Zelle je Spitze
    characteristics: tuple        # je Zelle: normierte Charakteristik ((Elektrode, relative Verzögerung), ...) mit Bezug auf die Kontaktelektrode mit kleinstem Index

    @property
    def n_samples(self):
        return self.X.shape[1]

    @property
    def nu(self):
        """Untere Schranke der Spikezahl je Zelle (Länge * untere Feuerrate), wie in der Dissertation aus der Aufnahmedauer abgeleitet."""
        return max(1, int(self.n_samples / C.SAMPLE_RATE * C.LOWER_RATE))


def electrode_positions(grid):
    """Zentren (0.5 + j_x, 0.5 + j_y), Elektrode j = grid * j_y + j_x (Zeile für Zeile)."""
    jj = np.arange(grid * grid)
    return np.column_stack([0.5 + jj % grid, 0.5 + jj // grid])


def neighbourhood_true(ds):
    """Die Menge der wahren Nachbarschaften (als frozensets)."""
    return {frozenset(int(j) for j in n) for n in ds.neighbourhoods}


def _template_params(rng, similarity):
    """Zufällige Parameter einer Vorlage (Intervalle der Dissertation); Ähnlichkeit 0 = alle Kontakte die Referenzform."""
    ref = C.REFERENCE_TEMPLATE
    a1 = rng.uniform(*C.ALPHA1_RANGE)
    drawn = dict(alpha1=a1, alpha2=-a1 * rng.uniform(*C.ALPHA2_FACTOR), beta1=rng.uniform(*C.BETA1_FRACTION), gamma1=rng.uniform(*C.GAMMA1_FRACTION), gamma2=rng.uniform(*C.GAMMA2_FRACTION))
    return {k: ref[k] + similarity * (drawn[k] - ref[k]) for k in ref}


def template(params):
    """Vorlagenwerte phi(0..w) und die Lage der Spitze (Zeitpunkt des Minimums)."""
    w = C.WIDTH
    t = np.arange(w + 1, dtype=float)
    b1, b2 = params["beta1"] * w, 0.5 * w
    phi = params["alpha1"] * np.exp(-((t - b1) ** 2) / (2 * (params["gamma1"] * w) ** 2)) + params["alpha2"] * np.exp(-((t - b2) ** 2) / (2 * (params["gamma2"] * w) ** 2))
    return phi, int(np.argmin(phi))


def firing_times(index, n_samples, seed, rate_scale):
    """Erneuerungsprozess: Abstand = Refraktärzeit + Exponentialverteilung, Rate der Zelle gleichverteilt in RATE_RANGE (mal Faktor). Feuerzeiten in Abtastwerten."""
    rng = np.random.default_rng([seed, index])
    rate = rng.uniform(*C.RATE_RANGE) * rate_scale
    mean_isi = C.SAMPLE_RATE / rate
    n_draw = int(n_samples / (mean_isi - C.REFRACTORY) * 1.5) + 20
    isi = C.REFRACTORY + rng.exponential(mean_isi - C.REFRACTORY, n_draw)
    times = np.cumsum(isi) - isi[0] + rng.uniform(0, mean_isi)
    return times[times < n_samples - C.WIDTH].astype(int)


def _place(series, peaks, phi, pi, scale=1.0):
    """Vorlage phi (Spitze bei Index pi) an den Spitzenzeiten `peaks` in die Spur `series` addieren."""
    T = len(series)
    impulses = np.zeros(T)
    ok = (peaks >= 0) & (peaks < T)
    np.add.at(impulses, peaks[ok], scale)
    series += np.convolve(impulses, phi)[pi: pi + T]


def _cell(index, seed, grid, contact_p, similarity, delay_max):
    """Ort, Kontaktelektroden und deren Verzögerungen und Vorlagen einer Zelle (Zellen ohne Kontakt werden neu gewürfelt)."""
    rng = np.random.default_rng([seed, 1000 + index])
    pos = electrode_positions(grid)
    for _ in range(10_000):
        cell = rng.uniform(0, grid, 2)
        dist = np.linalg.norm(pos - cell, axis=1)
        near = np.flatnonzero(dist <= C.RADIUS)
        contacts = near[rng.random(len(near)) < contact_p]
        if len(contacts):
            break
    else:
        contacts = np.array([int(np.argmin(dist))])
    rng2 = np.random.default_rng([seed, 2000 + index])
    delays = rng2.integers(0, delay_max + 1, len(contacts)) if delay_max > 0 else np.zeros(len(contacts), dtype=int)
    params = [_template_params(rng2, similarity) for _ in contacts]
    # pseudo-adjazente Elektroden: kein Kontakt, aber Abstand höchstens 2R
    pseudo = np.array([j for j in np.flatnonzero(dist <= 2 * C.RADIUS) if j not in set(contacts.tolist())], dtype=int)
    rng3 = np.random.default_rng([seed, 3000 + index])
    pseudo_delays = rng3.integers(0, delay_max + 1, len(pseudo)) if delay_max > 0 else np.zeros(len(pseudo), dtype=int)
    pseudo_params = [_template_params(rng3, similarity) for _ in pseudo]
    return cell, np.sort(contacts), delays[np.argsort(contacts)], [params[k] for k in np.argsort(contacts)], pseudo, pseudo_delays, pseudo_params


def make_dataset(n_neurons, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed):
    m, n = n_neurons, grid * grid
    T = int(round(seconds * C.SAMPLE_RATE))
    X_clean = np.zeros((n, T))
    S = np.zeros((m, T))
    cell_pos, nbhd, delays_all, fire_all, spikes_all, peaks_all, chars = [], [], [], [], [], [], []
    ep = [[] for _ in range(n)]
    epc = [[] for _ in range(n)]
    eps = [[] for _ in range(n)]
    for i in range(m):
        cell, contacts, delays, params, pseudo, pseudo_delays, pseudo_params = _cell(i, seed, grid, contact_p, similarity, delay_max)
        s = firing_times(i, T, seed, rate_scale)
        jrng = np.random.default_rng([seed, 4000 + i])
        peaks = np.zeros((len(s), len(contacts)), dtype=int)
        for c, (j, d, p) in enumerate(zip(contacts, delays, params)):
            jit = np.rint(jitter * jrng.standard_normal(len(s))).astype(int) if jitter > 0 else 0
            peaks[:, c] = s + d + jit
            phi, pi = template(p)
            _place(X_clean[j], peaks[:, c], phi, pi)
        for j, d, p in zip(pseudo, pseudo_delays, pseudo_params):
            phi, pi = template(p)
            _place(X_clean[j], s + d, phi, pi, C.BACKGROUND_ATTENUATION)
        strongest = int(np.argmin([q["alpha1"] for q in params]))
        phi, pi = template(params[strongest])
        _place(S[i], s + delays[strongest], phi, pi)
        S[i] = (S[i] - S[i].mean()) / max(S[i].std(), 1e-12)
        rel = delays - delays[0]                                   # Bezug: Kontaktelektrode mit kleinstem Index (contacts sind sortiert)
        chars.append(tuple((int(j), int(r)) for j, r in zip(contacts, rel)))
        cell_pos.append(cell)
        nbhd.append(contacts)
        delays_all.append(delays)
        fire_all.append(s)
        spikes_all.append(s + int(delays.min()))
        peaks_all.append(peaks)
        for c, j in enumerate(contacts):
            ep[j].append(peaks[:, c])
            epc[j].append(np.full(len(s), i))
            eps[j].append(np.arange(len(s)))
    electrode_peaks, electrode_cell, electrode_spike = [], [], []
    for j in range(n):
        if ep[j]:
            t, ci, si = np.concatenate(ep[j]), np.concatenate(epc[j]), np.concatenate(eps[j])
            order = np.argsort(t, kind="stable")
            electrode_peaks.append(t[order]), electrode_cell.append(ci[order]), electrode_spike.append(si[order])
        else:
            electrode_peaks.append(np.zeros(0, dtype=int)), electrode_cell.append(np.zeros(0, dtype=int)), electrode_spike.append(np.zeros(0, dtype=int))
    sigma = float(noise)
    X = X_clean + sigma * np.random.default_rng([seed, 9000]).standard_normal(X_clean.shape)
    return Dataset(X, X_clean, S, m, n, grid, seed, sigma, (n_neurons, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed), int(delay_max), float(jitter), electrode_positions(grid), np.array(cell_pos), tuple(nbhd), tuple(delays_all), tuple(fire_all), tuple(spikes_all), tuple(peaks_all),
                   tuple(electrode_peaks), tuple(electrode_cell), tuple(electrode_spike), tuple(chars))
