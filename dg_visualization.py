"""Plotly-Visualisierungen der Verzögerungsgraph-Demo: Elektrodengitter mit Zellen und Nachbarschaften, Elektrodenspuren, Spitzenschätzung, Zeitdifferenzen eines Elektrodenpaares, der Verzögerungsgraph im Zeitfenster,
Raster, Verwechslungsmatrix, Kennzahlen-Balken, Sweeps und Szenen-Vergleich. Alle Figuren laufen durch `lock_axes` (Touch-Scrolling-Konvention des Portfolios)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dg_constants as C
import dg_delay as dd

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE, TEAL, MAGENTA = "#1f77b4", "#d68a2e", "#2ca02c", "#d62728", "#8a8f98", "#8e5fbf", "#00838f", "#c2185b"
CLUSTER_COLORS = ("#1f77b4", "#d68a2e", "#2ca02c", "#8e5fbf", "#c2185b", "#00838f", "#6d4c41", "#455a64", "#9e9d24", "#e377c2", "#17becf", "#7f7f7f")
METHOD_COLORS = {"dg": MAGENTA, "pipe": BLUE, "tm": TEAL, "ica": ORANGE, "sca": GREEN}
METHOD_NAMES = {"dg": "Verzögerungsgraph", "pipe": "Pipeline", "tm": "Vorlagenabgleich", "ica": "ICA", "sca": "SCA"}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def cell_color(i):
    return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]


def _window(t0, width):
    fs = C.SAMPLE_RATE
    return int(t0 * fs / 1000), int((t0 + width) * fs / 1000)


def build_grid(ds, estimated=None):
    """Elektroden (Quadrate) und Zellen (Kreise, Farbe = Zelle) mit ihren wahren Kontakten (dünne Linien). Mit `estimated` (bestätigte Nachbarschaften): grüne dicke Linien = richtig gefundene Nachbarschaft,
    rote = falsch, orange gestrichelt = wahre Nachbarschaft, die fehlt; Ringe um Elektroden bei Nachbarschaften aus nur einer Elektrode."""
    pos = ds.positions
    fig = go.Figure()
    for i in range(ds.n_neurons):
        for j in ds.neighbourhoods[i]:
            fig.add_trace(go.Scatter(x=[ds.cell_positions[i][0], pos[j][0]], y=[ds.cell_positions[i][1], pos[j][1]], mode="lines", line=dict(color=cell_color(i), width=1), opacity=0.55, hoverinfo="skip", showlegend=False))
    if estimated is not None:
        true = {frozenset(int(j) for j in n) for n in ds.neighbourhoods}
        est = set(estimated)

        def draw(nb, color, width, dash):
            nb = sorted(nb)
            if len(nb) == 1:
                fig.add_trace(go.Scatter(x=[pos[nb[0]][0]], y=[pos[nb[0]][1]], mode="markers", marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=color, width=3)), hoverinfo="skip", showlegend=False))
            else:
                order = sorted(nb, key=lambda j: (pos[j][0], pos[j][1]))
                fig.add_trace(go.Scatter(x=[pos[j][0] for j in order], y=[pos[j][1] for j in order], mode="lines", line=dict(color=color, width=width, dash=dash), hoverinfo="skip", showlegend=False))
        for nb in true - est:
            draw(nb, ORANGE, 3, "dash")
        for nb in est:
            draw(nb, GREEN if nb in true else RED, 5, "solid")
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers+text", text=[f"E{j}" for j in range(len(pos))], textposition="bottom center", marker=dict(symbol="square", size=12, color=GRAY), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=ds.cell_positions[:, 0], y=ds.cell_positions[:, 1], mode="markers+text", text=[f"Z{i + 1}" for i in range(ds.n_neurons)], textposition="top center",
                             marker=dict(size=13, color=[cell_color(i) for i in range(ds.n_neurons)], line=dict(color="black", width=1)), hoverinfo="skip", showlegend=False))
    g = ds.grid
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[-0.3, g + 0.3], showgrid=False, zeroline=False, scaleanchor="y", constrain="domain", title="Ort (Elektrodenabstand = 1)"),
                      yaxis=dict(range=[g + 0.3, -0.3], showgrid=False, zeroline=False, constrain="domain", title=""))
    return lock_axes(fig)


def build_traces(labels, arrays, t0, width, peaks=None, estimated=None, colors=None, height=None):
    """Gestapelte Elektrodenspuren eines Zeitfensters [t0, t0 + width) in ms; ▼ = wahre Spitzen (nur zur Bewertung), Punkte = geschätzte Spitzen."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    n = len(arrays)
    scale = max(float(np.abs(a).max()) for a in arrays)
    fig = go.Figure()
    for r, (label, y) in enumerate(zip(labels, arrays)):
        offset = (n - 1 - r) * 1.3
        color = colors[r] if colors else BLUE
        fig.add_trace(go.Scatter(x=np.arange(lo, hi) * 1000.0 / fs, y=offset + y[lo:hi] / max(scale, 1e-12), mode="lines", line=dict(color=color, width=1.2), name=label, hoverinfo="skip"))
        if peaks is not None:
            marks = [t for t in peaks[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=[offset + 0.75] * len(marks), mode="markers", marker=dict(symbol="triangle-down", size=7, color=ORANGE), hoverinfo="skip", showlegend=False))
        if estimated is not None:
            marks = [t for t in estimated[r] if lo <= t < hi]
            if marks:
                fig.add_trace(go.Scatter(x=np.array(marks) * 1000.0 / fs, y=offset + y[np.array(marks)] / max(scale, 1e-12), mode="markers", marker=dict(color=GREEN, size=6), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height or max(200, 46 * n + 60), margin=dict(l=10, r=10, t=10, b=10), showlegend=False, xaxis=dict(title="Zeit [ms]"),
                      yaxis=dict(tickmode="array", tickvals=[(n - 1 - r) * 1.3 for r in range(n)], ticktext=list(labels), zeroline=False))
    return lock_axes(fig)


def build_peak_estimation(x, estimated, true_peaks, t0, width):
    """Spitzenschätzung einer Elektrode: oben Signal (grün = geschätzte, orange × = verpasste wahre Spitze), Mitte geglättet, unten die Differenzfolge mit den Schwellen eta- und eta+ (Dissertation Abschn. 2.2.2)."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    smooth, diff, eta_minus, eta_plus = dd.peak_signals(x)
    t = np.arange(lo, hi) * 1000.0 / fs
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=("Signal", "geglättet (Binomialfilter)", "Differenz r(t) − r(t−d), geglättet"))
    fig.add_trace(go.Scatter(x=t, y=x[lo:hi], mode="lines", line=dict(color=GRAY, width=1), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=smooth[lo:hi], mode="lines", line=dict(color=BLUE, width=1.6), hoverinfo="skip", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=diff[lo:hi], mode="lines", line=dict(color=PURPLE, width=1.6), hoverinfo="skip", showlegend=False), row=3, col=1)
    fig.add_hline(y=eta_minus, line=dict(color=RED, dash="dash"), row=3, col=1)
    fig.add_hline(y=eta_plus, line=dict(color=GREEN, dash="dash"), row=3, col=1)
    est = np.asarray(estimated)
    inside = est[(est >= lo) & (est < hi)]
    if len(inside):
        for row, series in ((1, x), (2, smooth)):
            fig.add_trace(go.Scatter(x=inside * 1000.0 / fs, y=series[inside], mode="markers", marker=dict(color=GREEN, size=8), hoverinfo="skip", showlegend=False), row=row, col=1)
    tp = np.asarray(true_peaks)
    tp = tp[(tp >= lo) & (tp < hi)]
    missed = [int(p) for p in tp if len(est) == 0 or np.abs(est - p).min() > 6]
    if missed:
        fig.add_trace(go.Scatter(x=np.array(missed) * 1000.0 / fs, y=x[missed], mode="markers", marker=dict(color=ORANGE, size=10, symbol="x"), hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.update_xaxes(title="Zeit [ms]", row=3, col=1)
    fig.update_yaxes(title_text="µV")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10))
    return lock_axes(fig)


def build_differences(diffs, mask, delta, title):
    """Zeitdifferenzen aller Spitzenpaare zweier Elektroden (höchstens delta): grün die zulässigen (in einem Fenster mit mindestens (1 − θ1)·ν Vorkommen), grau die zufälligen."""
    values = np.arange(-delta, delta + 1)
    good = np.array([int(((diffs == v) & mask).sum()) for v in values])
    bad = np.array([int(((diffs == v) & ~mask).sum()) for v in values])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=values, y=bad, name="zufällig", marker_color=GRAY, hoverinfo="skip"))
    fig.add_trace(go.Bar(x=values, y=good, name="zulässig (stabil)", marker_color=GREEN, hoverinfo="skip"))
    fig.update_layout(height=300, barmode="stack", margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title=f"Zeitdifferenz {title} [Abtastwerte]"), yaxis=dict(title="Anzahl der Paare"), legend=dict(orientation="h", y=-0.3))
    return lock_axes(fig)


def build_delay_graph(graph, sorting, t0, width, max_nodes=400):
    """Der geschätzte Verzögerungsgraph im Zeitfenster: Knoten = geschätzte Spitzen (x = Zeit, y = Elektrode), Kanten = zulässige Zeitdifferenzen; Farbe = Neuron der Clique, grau = keiner bestätigten Charakteristik zugeordnet."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    color_of = {}
    for k, clique in enumerate(sorting.cliques):
        for node in clique:
            color_of.setdefault(node, cell_color(int(sorting.labels[k])))
    nodes = [(j, l) for j in range(len(graph.peaks)) for l, t in enumerate(graph.peaks[j]) if lo <= t < hi][:max_nodes]
    inside = set(nodes)
    fig = go.Figure()
    seen = set()
    for a in nodes:
        for j2, ls in graph.adjacency[a].items():
            for l2 in ls:
                b = (j2, l2)
                if b in inside and (b, a) not in seen:
                    seen.add((a, b))
                    fig.add_trace(go.Scatter(x=[graph.peaks[a[0]][a[1]] * 1000.0 / fs, graph.peaks[b[0]][b[1]] * 1000.0 / fs], y=[a[0], b[0]], mode="lines", line=dict(color=color_of.get(a, GRAY), width=1.5), opacity=0.7,
                                             hoverinfo="skip", showlegend=False))
    if nodes:
        fig.add_trace(go.Scatter(x=[graph.peaks[j][l] * 1000.0 / fs for j, l in nodes], y=[j for j, _ in nodes], mode="markers", marker=dict(size=8, color=[color_of.get(nd, GRAY) for nd in nodes], line=dict(color="black", width=0.5)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]", range=[lo * 1000.0 / fs, hi * 1000.0 / fs]), yaxis=dict(title="Elektrode", dtick=1, autorange="reversed"))
    return lock_axes(fig)


def build_confusion(confusion, cluster_of_neuron, label="Charakteristik"):
    """Verwechslungsmatrix: Zeilen = wahre Zellen, Spalten = gefundene Neuronen (nach der optimalen Zuordnung sortiert, nicht zugeordnete hinten); die Diagonale ist die richtige Sortierung."""
    m, k = confusion.shape
    order = [c for c in cluster_of_neuron if c >= 0] + [c for c in range(k) if c not in cluster_of_neuron]
    Z = confusion[:, order]
    fig = go.Figure(go.Heatmap(z=Z, x=[f"{label} {c + 1}" for c in order], y=[f"Zelle {i + 1}" for i in range(m)], colorscale="Blues", text=Z.astype(int), texttemplate="%{text}", showscale=False, hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=80 + 24 * m, margin=dict(l=10, r=10, t=10, b=10))
    return lock_axes(fig)


def build_raster(truth_times, truth_cell, detected, neuron_of_event, t0, width, n_cells):
    """Spikes im Zeitfenster: je Zelle eine Zeile mit den wahren Spikes (oben, Strich) und den ihr zugeordneten Ereignissen (unten, Punkt); falsch zugeordnete liegen in der Zeile der falschen Zelle."""
    fs = C.SAMPLE_RATE
    lo, hi = _window(t0, width)
    fig = go.Figure()
    for i in range(n_cells):
        base = (n_cells - 1 - i) * 1.0
        tsel = truth_times[(truth_cell == i) & (truth_times >= lo) & (truth_times < hi)]
        dsel = detected[(neuron_of_event == i) & (detected >= lo) & (detected < hi)]
        if len(tsel):
            fig.add_trace(go.Scatter(x=tsel * 1000.0 / fs, y=np.full(len(tsel), base + 0.25), mode="markers", marker=dict(symbol="line-ns-open", size=12, color=cell_color(i), line=dict(width=2)), hoverinfo="skip", showlegend=False))
        if len(dsel):
            fig.add_trace(go.Scatter(x=dsel * 1000.0 / fs, y=np.full(len(dsel), base - 0.05), mode="markers", marker=dict(symbol="circle", size=7, color=cell_color(i)), hoverinfo="skip", showlegend=False))
    fig.update_layout(height=80 + 26 * n_cells, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(title="Zeit [ms]", range=[lo * 1000.0 / fs, hi * 1000.0 / fs]),
                      yaxis=dict(tickmode="array", tickvals=[(n_cells - 1 - i) * 1.0 + 0.1 for i in range(n_cells)], ticktext=[f"Z{i + 1}" for i in range(n_cells)], zeroline=False))
    return lock_axes(fig)


def build_method_bars(analysis):
    """Spitzen-F1 der Zellen: Verzögerungsgraph, Pipeline, Vorlagenabgleich, ICA und (bei mehr Zellen als Elektroden) SCA."""
    values = {"dg": analysis.dg.f1, "pipe": analysis.pipe.f1, "tm": analysis.tm.f1, **{k: v["f1"] for k, v in analysis.comparators.items()}}
    names = [k for k in ("dg", "pipe", "tm", "ica", "sca") if k in values]
    fig = go.Figure(go.Bar(x=[METHOD_NAMES[n] for n in names], y=[values[n] for n in names], marker_color=[METHOD_COLORS[n] for n in names], text=[f"{values[n]:.2f}" for n in names], textposition="outside", hoverinfo="skip"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Zellen", range=[0, 1.15]), showlegend=False)
    return lock_axes(fig)


def build_sweep(rows, xlabel, current=None):
    """Links: Spitzen-F1 aller Verfahren (Band = Spanne des Verzögerungsgraphen über die Datensätze); rechts: Trefferquote, Genauigkeit der Detektion, Sortiergenauigkeit des Verzögerungsgraphen und Anteil der Zellen mit nur einem Kontakt."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spitzen-F1", "Verzögerungsgraph im Detail"), horizontal_spacing=0.12)
    xs = [r["x"] for r in rows]
    lo, hi = [r["dg_min"] for r in rows], [r["dg_max"] for r in rows]
    fig.add_trace(go.Scatter(x=xs + xs[::-1], y=hi + lo[::-1], fill="toself", fillcolor=MAGENTA, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False), row=1, col=1)
    for name, dash in (("dg", "solid"), ("pipe", "solid"), ("tm", "solid"), ("ica", "solid")):
        ys = [r[name] for r in rows]
        if not all(np.isnan(ys)):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=METHOD_NAMES[name], line=dict(color=METHOD_COLORS[name], width=3 if name == "dg" else 1.8, dash=dash), hoverinfo="skip"), row=1, col=1)
    for key, name, color, dash in (("recall", "Trefferquote", GREEN, "solid"), ("precision", "Genauigkeit der Detektion", RED, "solid"), ("accuracy", "Sortiergenauigkeit", MAGENTA, "dash"), ("single_contact", "Anteil Einzelkontakt-Zellen", GRAY, "dot")):
        fig.add_trace(go.Scatter(x=xs, y=[r[key] for r in rows], mode="lines+markers", name=name, line=dict(color=color, width=2, dash=dash), hoverinfo="skip"), row=1, col=2)
    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(range=[0, 1.05])
    if current is not None:
        for col in (1, 2):
            fig.add_vline(x=current, line=dict(color=RED, dash="dash"), row=1, col=col)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.35))
    return lock_axes(fig)


def build_scenes(rows):
    """Verzögerungsgraph, Pipeline, Vorlagenabgleich und ICA: Spitzen-F1 je Szene; Balken = Mittel, Fehlerbalken = Spanne über die Sweep-Datensätze."""
    labels = [r["scene"].replace(" (", "<br>(") for r in rows]
    fig = go.Figure()
    for name in ("dg", "pipe", "tm", "ica"):
        y = [r[name] for r in rows]
        fig.add_trace(go.Bar(x=labels, y=y, name=METHOD_NAMES[name], marker_color=METHOD_COLORS[name], text=[f"{v:.2f}" for v in y], textposition="outside", hoverinfo="skip",
                             error_y=dict(type="data", symmetric=False, array=[r[name + "_max"] - r[name] for r in rows], arrayminus=[r[name] - r[name + "_min"] for r in rows])))
    fig.update_layout(height=520, barmode="group", margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(title="Spitzen-F1 der Zellen", range=[0, 1.2]), legend=dict(orientation="h", y=-0.6))
    return lock_axes(fig)
