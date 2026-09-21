"""Verzögerungsgraph (Zellen an Zeitverzögerungen erkennen statt an der Wellenform) - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - den Verzögerungsgraph aus der Dissertation des Autors - und lässt stattdessen das Beispiel wachsen.
Sechstes Stück der Quellentrennung-Linie der "Konzepte"-Reihe: ein eigener Nachbau des Verfahrens, verglichen mit der Standardpipeline, dem Vorlagenabgleich, ICA und SCA - mit Siegen und Niederlagen.
Der Vergleich ist neu und steht nicht in der Dissertation. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import streamlit as st

import dg_constants as C
from dg_evaluation import SWEEP_LABELS, Settings, analyse_for, make_dataset, scene_table, sweep, truth_spikes, verdict
from dg_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from dg_visualization import (
    build_confusion,
    build_delay_graph,
    build_differences,
    build_grid,
    build_method_bars,
    build_peak_estimation,
    build_raster,
    build_scenes,
    build_sweep,
    build_traces,
)

st.set_page_config(page_title="Verzögerungsgraph – Sebastian Hanisch", layout="wide")

STEP_LABELS = {1: "1 · Aufnahme", 2: "2 · Spitzen schätzen", 3: "3 · Verzögerungsgraph", 4: "4 · Nachbarschaften", 5: "5 · Zellen zuordnen"}
WINDOW_WIDTH_MS = 50
SWEEP_OPTIONS = {"n_cells": "Anzahl Zellen", "grid": "Gittergröße", "contact_p": "Kontaktwahrscheinlichkeit", "similarity": "Wellenform-Ähnlichkeit", "delay_max": "Verzögerungsspanne", "jitter": "Verzögerungs-Jitter",
                 "rate_scale": "Feuerrate", "noise": "Rauschen", "seconds": "Länge der Aufnahme"}


def _pct(x):
    return "–" if np.isnan(x) else f"{x:.0%}"


# Eine Analyse belegt ~134 MiB, ein Datensatz ~57 MiB (gemessen im Standardfall) - ohne Obergrenze füllt jede neue
# Reglerstellung den Speicher (Streamlit Cloud, CI-Rechner). Die neuesten Einträge bleiben, ältere werden verdrängt.
@st.cache_data(show_spinner=False, max_entries=4)
def _dataset(m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed):
    return make_dataset(m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds, seed)


@st.cache_data(show_spinner=False, max_entries=3)
def _analysis(data_params, settings):
    return analyse_for(data_params, settings)


@st.cache_data(show_spinner=False)
def _sweep(parameter, base, settings):
    m, grid, contact_p, similarity, delay_max, jitter, rate_scale, noise, seconds = base
    return sweep(parameter, settings=settings, m=m, grid=grid, contact_p=contact_p, similarity=similarity, delay_max=delay_max, jitter=jitter, rate_scale=rate_scale, noise=noise, seconds=seconds)


@st.cache_data(show_spinner=False)
def _scenes(base, settings):
    return scene_table(settings, seconds=base[8], noise=base[7])


st.title("🕸️ Verzögerungsgraph – Zellen an ihren Zeitverzögerungen erkennen")
st.markdown(
    """
Die bisherigen Spike-Sorting-Stücke erkennen Zellen an ihrer **Wellenform**: Pipeline, Vorlagenabgleich, ICA. Dieses Stück zeigt einen anderen Ansatz - den des Autors aus seiner Dissertation (Universität Rostock, 2017): Eine Zelle
feuert, und ihr Signal erreicht **verschiedene Elektroden mit jeweils festen zeitlichen Verzögerungen**. Der **Verzögerungsgraph** verbindet Spitzen verschiedener Elektroden, deren Zeitabstand sich **immer wieder** genauso zeigt;
zusammenhängende Gruppen solcher Spitzen (**Cliquen**) sind Spikes derselben Zelle, die Menge ihrer Elektroden ist ihre **Nachbarschaft**. Das Verfahren braucht **keine Wellenform** - nur die Zeitpunkte der Spitzen.
Wie das Verfahren funktioniert, erklärt der aufgeklappte Abschnitt direkt darunter.
"""
)
st.info(
    "**Eigener Nachbau, Vergleich neu.** Das Verfahren ist hier von Grund auf nach der Dissertation nachgebaut; die Dissertation selbst vergleicht es nicht mit Wellenform-Verfahren. Der Vergleich mit den anderen vier Verfahren ist neu, "
    "läuft auf synthetischen Daten und auf **dem Modell aus der Dissertation** - also dem Terrain, für das das Verfahren gedacht ist. Deshalb zeigen Szenen und Presets auch, wo es verliert."
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - sechstes Stück der Quellentrennung-Linie der \"Konzepte\"-Reihe - **ein** Verfahren an einem wachsenden Beispiel."
)

with st.expander("So funktioniert der Verzögerungsgraph", expanded=True):
    st.markdown(
        """
1. **Spitzen schätzen.** Je Elektrode wird das Signal geglättet, die Differenz $r(t) - r(t-d)$ gebildet und nach einem steilen Abstieg mit folgendem Anstieg gesucht; das Minimum dazwischen ist die geschätzte **Spitze** (Zeitpunkt, keine Form).
2. **Verzögerungsgraph.** Jede Spitze ist ein Knoten. Zwei Spitzen **verschiedener** Elektroden (nicht weiter als $2R$ voneinander entfernt) werden verbunden, wenn ihr Zeitabstand **zulässig** ist: Er liegt in einem kurzen Fenster,
   in dem (fast) so viele Abstände liegen, wie die Zelle mindestens gefeuert haben muss ($\\nu$). Zufällige Nachbarschaft von Spitzen erzeugt kein solches Häufungsfenster - die feste Verzögerung einer Zelle schon.
3. **Nachbarschaften.** Die Elektroden, mit denen ein Knoten verbunden ist, bilden seine Nachbarschaft; Mengen, die oft genug vorkommen, sind die Nachbarschaften der Zellen.
4. **Zellen zuordnen.** Aus den Nachbarschaftsgraphen wird per **Greedy-Clique** je Spike eine Gruppe zusammengehöriger Spitzen gesucht. Ihre **Charakteristik** (Elektroden und Zeitabstände zur ersten Elektrode) kennzeichnet die Zelle;
   Charakteristiken, die mindestens $\\nu/2$ Mal vorkommen, sind Zellen, jede zugeordnete Clique ist ein Spike dieser Zelle. Ist eine Clique nur ein einzelner Knoten, erkennt das Verfahren die Zelle allein an ihrer Elektrode
   (Zelle mit **einem** Kontakt - die Dissertation lässt das in ihrer Heuristik zu; hier ein Schalter).
5. **Bewertung** (nur hier möglich, weil die Wahrheit bekannt ist): Ereignisse werden den wahren Spikes zugeordnet (Zeitfenster ±0.3 ms), Zellen und Charakteristiken optimal einander zugeordnet - für **alle** Verfahren dieselbe Auswertung.

Was das Verfahren **verlangt**: Zellen, die über mehrere Elektroden mit stabilen Verzögerungen gesehen werden, und genug Spikes. Was es **nicht** verlangt: eine feste Wellenform, ein Mischungsmodell oder weniger Zellen als Elektroden.
Was es **nicht kann**: Zellen unterscheiden, die dieselbe normierte Charakteristik haben (Proposition 2.3.1 der Dissertation) - zwei Zellen mit denselben Elektroden und denselben relativen Verzögerungen.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_cells = st.slider(
        "Zellen", *bounds("n_cells_slider"), key="n_cells_slider",
        help="Zellen mit zufälligem Ort auf dem Elektrodengitter. Spitzen-F1 des Verzögerungsgraphen (Pipeline / Abgleich / ICA) bei 25 Elektroden: 6 Zellen 0.99 (0.93 / 0.95 / 0.85), 12 Zellen 0.97 (0.73 / 0.85 / 0.72), "
             "25 Zellen 0.95 (0.52 / 0.76 / 0.55), 30 Zellen 0.94 (0.47 / 0.74 / 0.50) - mehr Zellen als Elektroden bringt ICA und Pipeline ins Rutschen, den Verzögerungsgraphen kaum.",
    )
    grid = st.slider(
        "Gitter (Kantenlänge)", *bounds("grid_slider"), key="grid_slider",
        help="Elektroden liegen auf einem Gitter Kantenlänge × Kantenlänge, Abstand 1; Zellen haben Kontakte zu Elektroden im Radius 2. Bei 12 Zellen: 3×3 Elektroden 0.86 (Pipeline / Abgleich / ICA: 0.76 / 0.92 / 0.42), "
             "4×4 0.91 (0.74 / 0.82 / 0.64), 5×5 0.97, 6×6 0.99 (0.78 / 0.85 / 0.79). Auf dem kleinen Gitter teilen sich mehrere Zellen dieselben Elektroden - der Abgleich gewinnt dort.",
    )
    contact_p = st.slider(
        "Kontaktwahrscheinlichkeit p", *bounds("contact_p_slider"), key="contact_p_slider", step=0.05,
        help="Wahrscheinlichkeit, dass eine Zelle eine Elektrode in ihrem Radius wirklich berührt (Dissertation: 0.2). Mit p = 0.1 haben 63 % der Zellen nur einen Kontakt: Spitzen-F1 0.86 (Pipeline 0.71, Abgleich 0.74, ICA 0.71); "
             "ohne Zulassen von Einzelkontakt-Zellen nur 0.37. Ab p = 0.35 ist der Verzögerungsgraph nahe 1 (0.97 bei 0.35, 0.98 bei 0.5).",
    )
    similarity = st.slider(
        "Wellenform-Ähnlichkeit", *bounds("similarity_slider"), key="similarity_slider", step=0.05,
        help="1 = Vorlagen wie in der Dissertation (zufällige Parameter je Kontakt), 0 = alle Kontakte aller Zellen dieselbe Form und Amplitude. Der Verzögerungsgraph sieht die Form nie: 0.99 bei Ähnlichkeit 0 und 0.97 bei 1; "
             "die Pipeline fällt von 0.73 auf 0.67, der Abgleich von 0.85 auf 0.81. Die Zellen unterscheiden sich dann nur noch über ihre Elektroden und Verzögerungen.",
    )
    delay_max = st.slider(
        "Verzögerungsspanne [Abtastwerte]", *bounds("delay_max_slider"), key="delay_max_slider",
        help="Obere Schranke der Verzögerungen zwischen Zelle und Elektrode (Dissertation: 4 Abtastwerte = 0.1 ms bei 40 kHz). Der Verzögerungsgraph ist davon unabhängig (0.99 bei 0 und bei 8: die Elektrodenmengen unterscheiden die Zellen schon); "
             "ICA leidet, weil die Mischung nicht mehr augenblicklich ist (0.77 bei 0, 0.62 bei 8).",
    )
    jitter = st.slider(
        "Verzögerungs-Jitter [Abtastwerte]", *bounds("jitter_slider"), key="jitter_slider", step=0.5,
        help="Zufällige Abweichung der Verzögerung von Spike zu Spike (Standardabweichung) - verletzt die Annahme konstanter Verzögerungen, auf der der Verzögerungsgraph beruht (Dissertation: keine). "
             "Spitzen-F1 des Verzögerungsgraphen 0.97 (0), 0.91 (1), 0.70 (2), 0.62 (3); der Vorlagenabgleich hält 0.85 / 0.80 / 0.77 / 0.63 - bei Jitter 2 gewinnt er.",
    )
    rate_scale = st.slider(
        "Feuerrate (Faktor)", *bounds("rate_slider"), key="rate_slider", step=0.25,
        help="Faktor auf die Feuerraten (5-30 Hz). Der Verzögerungsgraph nimmt an, dass jede Zelle mindestens 5 Hz feuert (Dissertation: untere Schranke ν): Faktor 0.25 unterschreitet das - Spitzen-F1 0.72 (Pipeline 0.86, Abgleich 0.82). "
             "Mehr Feuern hilft ihm eher: 0.96 bei Faktor 4, wo die Pipeline auf 0.48 fällt.",
    )
    noise = st.slider(
        "Rauschen [µV]", *bounds("noise_slider"), key="noise_slider", step=1.0,
        help="Standardabweichung des weißen Rauschens (Dissertation: 10 µV, Spitzenhöhe 50-200 µV). Der Verzögerungsgraph ist robust, solange die Spitzen erkennbar sind: 0.88 bei 20 µV (Pipeline 0.58, Abgleich 0.69), "
             "0.72 bei 30 µV (0.27 / 0.36), 0.60 bei 40 µV (0.10 / 0.11). Die Nachbarschafts-Fehler der Dissertation steigen dabei stark (bei 20 µV fehlen 20 % und 45 % sind falsch).",
    )
    seconds = st.slider(
        "Länge der Aufnahme [s]", *bounds("seconds_slider"), key="seconds_slider", step=0.5,
        help="Länge bei 40 kHz. ν (die Mindestzahl der Spikes je Zelle) wird aus der Länge und der unteren Feuerrate 5 Hz abgeleitet. Schon 1 s genügt: 0.99 (Pipeline 0.72, Abgleich 0.73).",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**Verzögerungsgraph**")
    singles = st.selectbox(
        "Zellen mit nur einem Kontakt", C.SINGLE_MODES, key="singles_select", format_func=lambda s: C.SINGLE_LABELS[s],
        help="Eine Zelle mit nur einer Elektrode hat keine Verzögerung zu einer zweiten - die Dissertation nimmt in ihrer Heuristik trotzdem Cliquen aus einem einzigen Knoten zu: eine Spitze ohne bestätigte Kante gilt als Spike einer Zelle "
             "mit dieser einen Elektrode (Standardfall: Spitzen-F1 0.97). Ausgeschlossen (nur Spikes mit bestätigter Kante) fallen diese Zellen weg: 0.77 im Standardfall, 0.37 bei p = 0.1. "
             "Zugelassen können sich mehrere Zellen auf derselben Elektrode zu einer vermischen (Prop. 2.3.1).",
    )

    st.button("🎲 Neue Aufnahme generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für Zellorte, Spikezeiten und Rauschen.")

sync_query_params({
    "n_cells_slider": int(n_cells), "grid_slider": int(grid), "contact_p_slider": float(contact_p), "similarity_slider": float(similarity), "delay_max_slider": int(delay_max), "jitter_slider": float(jitter),
    "rate_slider": float(rate_scale), "noise_slider": float(noise), "seconds_slider": float(seconds), "singles_select": singles, "seed_input": int(seed),
})

data_params = (int(n_cells), int(grid), float(round(contact_p, 2)), float(round(similarity, 2)), int(delay_max), float(round(jitter, 1)), float(rate_scale), float(noise), float(seconds), int(seed))
settings = Settings(min_clique=C.SINGLE_CLIQUE[singles])
with st.spinner("Analysiere die Aufnahme (Verzögerungsgraph, Pipeline, Vorlagenabgleich, ICA)..."):
    ds = _dataset(*data_params)
    analysis = _analysis(data_params, settings)
    level, code, vd = verdict(analysis)
delay, dg, pipe, tm_res = analysis.delay, analysis.dg, analysis.pipe, analysis.tm
graph, sorting = delay.graph, delay.sorting
m_n, n_el = ds.n_neurons, ds.n_electrodes
data_key = data_params + (settings,)
truth_times, truth_cell, truth_collision = truth_spikes(ds)
busiest = np.argsort([-len(p) for p in ds.electrode_peaks])[:6]
busiest = sorted(int(j) for j in busiest)
star = int(np.argmax([len(p) for p in ds.electrode_peaks]))

# --- Verzögerungsgraph in Aktion ------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Verzögerungsgraph in Aktion")
if "dg_step" not in st.session_state or st.session_state.get("dg_step_owner") != data_key:
    st.session_state["dg_step"] = 1
    st.session_state["dg_step_owner"] = data_key
duration_ms = ds.n_samples * 1000.0 / C.SAMPLE_RATE
max_start = int(duration_ms - WINDOW_WIDTH_MS)
if st.session_state.get("window_start", 0) > max_start:
    st.session_state["window_start"] = 0
step_col, play_col, win_col = st.columns([4, 2, 3])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="dg_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
with win_col:
    window = st.slider(f"Zeitfenster ({WINDOW_WIDTH_MS} ms) ab [ms]", 0, max_start, key="window_start", step=10, help="Welchen Ausschnitt der Aufnahme die Zeitreihen zeigen.")
view_slot = st.empty()


def _pair_choice():
    """Ein Elektrodenpaar mit vielen zulässigen Differenzen (gemeinsame Zelle) und eines ohne (nur Zufall)."""
    best, worst = None, None
    for pair, (diffs, mask) in graph.pair_differences.items():
        if best is None or mask.sum() > graph.pair_differences[best][1].sum():
            best = pair
        if mask.sum() == 0 and len(diffs) > 0 and (worst is None or len(diffs) > len(graph.pair_differences[worst][0])):
            worst = pair
    return best, worst


def _render(current_step):
    with view_slot.container():
        if current_step == 1:
            c1, c2 = st.columns([2, 3])
            c1.markdown("**Elektroden (Quadrate), Zellen (Kreise) und ihre wahren Kontakte** (unbekannt in der Praxis)")
            c1.plotly_chart(build_grid(ds), width="stretch", key="step_grid")
            c2.markdown(f"**Die sechs aktivsten Elektroden** (▼ = wahre Spitzen der Zellen, {len(busiest)} von {n_el})")
            c2.plotly_chart(build_traces([f"E{j}" for j in busiest], [ds.X[j] for j in busiest], window, WINDOW_WIDTH_MS, peaks=[ds.electrode_peaks[j] for j in busiest]), width="stretch", key="step_electrodes")
        elif current_step == 2:
            st.markdown(f"**Spitzenschätzung an Elektrode E{star}** (grün: geschätzte Spitze, orange ×: verpasste wahre Spitze; Schwellen rot und grün gestrichelt)")
            st.plotly_chart(build_peak_estimation(ds.X[star], graph.peaks[star], ds.electrode_peaks[star], window, WINDOW_WIDTH_MS), width="stretch", key="step_peaks")
        elif current_step == 3:
            best, worst = _pair_choice()
            c1, c2 = st.columns([2, 3])
            if best is not None:
                d, mk = graph.pair_differences[best]
                c1.markdown(f"**Zeitdifferenzen der Spitzen von E{best[0]} und E{best[1]}** (Paar mit gemeinsamer Zelle)")
                c1.plotly_chart(build_differences(d, mk, graph.delta, f"E{best[1]} − E{best[0]}"), width="stretch", key="step_diff_best")
            if worst is not None:
                d, mk = graph.pair_differences[worst]
                c1.markdown(f"**… und von E{worst[0]} und E{worst[1]}** (Paar ohne gemeinsame Zelle: nur Zufall)")
                c1.plotly_chart(build_differences(d, mk, graph.delta, f"E{worst[1]} − E{worst[0]}"), width="stretch", key="step_diff_worst")
            c2.markdown("**Der Verzögerungsgraph im Zeitfenster** (Knoten = Spitzen, x = Zeit, y = Elektrode; Farbe = Zelle der Clique)")
            c2.plotly_chart(build_delay_graph(graph, sorting, window, WINDOW_WIDTH_MS), width="stretch", key="step_graph")
        elif current_step == 4:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Geschätzte Nachbarschaften** (grün: richtig, rot: falsch, orange gestrichelt: fehlt; Ringe: Nachbarschaft aus einer Elektrode)")
            c1.plotly_chart(build_grid(ds, delay.neighbourhoods), width="stretch", key="step_neighbourhoods")
            c2.markdown("**Fehler wie in der Dissertation**")
            c2.metric("fehlende Nachbarschaften η_m", f"{analysis.eta_m:.0%}", help="Anteil der wahren Nachbarschaften, die nicht bestätigt wurden.")
            c2.metric("falsche Nachbarschaften η_f", f"{analysis.eta_f:.0%}", help="Zahl der bestätigten Nachbarschaften, die es nicht gibt, relativ zur Zahl der wahren.")
        else:
            c1, c2 = st.columns([3, 2])
            c1.markdown("**Raster: wahre Spikes (Striche) und ihre Zuordnung durch den Verzögerungsgraphen (Punkte)**")
            c1.plotly_chart(build_raster(truth_times, truth_cell, sorting.times, dg.neuron_of_event, window, WINDOW_WIDTH_MS, m_n), width="stretch", key="step_raster")
            c2.markdown("**Verwechslungsmatrix** (Zeilen: wahre Zellen, Spalten: gefundene Zellen)")
            c2.plotly_chart(build_confusion(dg.confusion, dg.cluster_of_neuron), width="stretch", key="step_confusion")


if auto_play:
    for s in STEP_LABELS:
        _render(s)
        time.sleep(1.2)
    step = 5
else:
    _render(step)

n_est = sum(len(p) for p in graph.peaks)
n_true = sum(len(p) for p in ds.electrode_peaks)
if step == 1:
    st.caption(f"{m_n} Zellen feuern (insgesamt {dg.n_true} Spikes) auf {n_el} Elektroden; jede Zelle hat Kontakt zu {np.mean([len(n) for n in ds.neighbourhoods]):.1f} Elektroden im Mittel, "
               f"{ds.n_neurons - sum(len(n) > 1 for n in ds.neighbourhoods)} Zelle(n) nur zu einer. Die Spitzen derselben Zelle erscheinen an ihren Elektroden mit festen Verzögerungen (bis {ds.delay_max} Abtastwerte = {ds.delay_max / 40:.2f} ms). "
               "Das Verfahren sieht nur die Elektrodensignale - Zellen, Kontakte und wahre Spitzen stehen nur zur Bewertung da.")
elif step == 2:
    st.caption(f"Insgesamt {n_est} geschätzte Spitzen auf allen Elektroden gegen {n_true} wahre. Geschätzt wird nur der **Zeitpunkt** - die Form der Spitze wird nie benutzt. Die Schwellen sind die der Dissertation (bei mehr Rauschen als 10 µV angehoben); "
               "bei hohem Rauschen gehen Spitzen verloren oder es entstehen zusätzliche.")
elif step == 3:
    st.caption(f"Bei zwei Elektroden, die dieselbe Zelle sehen, zeigt das Histogramm der Zeitdifferenzen **einen Gipfel** (die feste Relativverzögerung), bei zwei Elektroden ohne gemeinsame Zelle nur verstreute Werte. Grün ist, was in einem "
               f"Fenster von 4 Abtastwerten mit mindestens {(1 - C.THETA1) * graph.nu:.0f} Vorkommen liegt (ν = {graph.nu} aus der Länge und 5 Hz). Der Graph hat {graph.n_edges} Kanten. Jede Farbe im Graphen ist eine Zelle; "
               "graue Knoten gehören zu keiner bestätigten Charakteristik.")
elif step == 4:
    st.caption(f"Nachbarschaften, die oft genug als Elektrodenmenge eines Knotens vorkommen, gelten als bestätigt: {len(delay.neighbourhoods)} bestätigt, {len(ds.neighbourhoods)} Zellen (verschiedene Nachbarschaften: "
               f"{len({tuple(n) for n in ds.neighbourhoods})}). Zellen mit gleicher Elektrodenmenge fallen zu einer Nachbarschaft zusammen; Einzelelektroden erscheinen als Ringe.")
else:
    st.caption(f"{len(sorting.candidates)} Charakteristiken (Zellen) gefunden, {len(sorting.times)} Spikes zugeordnet ({sorting.n_singletons} von {sorting.n_cliques} Cliquen bestehen aus einem einzigen Knoten). "
               f"Trefferquote {dg.recall:.0%}, Genauigkeit der Detektion {dg.precision:.0%}, Sortiergenauigkeit {dg.accuracy:.0%} (nicht überlappende Spikes {_pct(dg.accuracy_single)}, überlappende {_pct(dg.accuracy_collision)}); "
               f"ein Klassifikator, der immer die häufigste Zelle nennt, läge bei {dg.majority_baseline:.0%}.")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was der Verzögerungsgraph gefunden hat - im Vergleich mit Pipeline, Vorlagenabgleich, ICA und SCA")
st.caption(
    "Spitzen-F1: je Zelle die Ereignisse ihres Neurons (bzw. die Spitzen ihrer geschätzten Spur bei ICA und SCA) gegen die wahren Spitzenzeiten (Toleranz ±0.3 ms), gemittelt über die Zellen - für alle Verfahren dieselbe Definition. "
    "Die Wellenform-Verfahren bekommen die wahre Zellzahl, die Pipeline so viele Hauptkomponenten wie Zellen (höchstens 15; bei drei wie in den Vorgängern ist sie für ein Dutzend Zellen zu schwach: 0.62 statt 0.73). "
    "ICA und SCA nehmen augenblickliche Mischung an, die es hier nicht gibt; SCA läuft nur, wenn es mehr Zellen als Elektroden gibt (ihr Anwendungsbereich)."
)
best_other = vd["best"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Spitzen-F1 (Verzögerungsgraph)", f"{dg.f1:.2f}", delta=f"{dg.f1 - best_other:+.2f} ggü. bestem anderen ({vd['best_name'].upper() if vd['best_name'] in ('ica', 'sca') else {'pipe': 'Pipeline', 'tm': 'Abgleich'}[vd['best_name']]})",
          delta_color="normal", help="Mittlerer Spitzen-F1 der Zellen; im Delta der Abstand zum besten der anderen Verfahren.")
m2.metric("Trefferquote", f"{dg.recall:.0%}", delta=f"Genauigkeit der Detektion {dg.precision:.0%}", delta_color="off", help="Anteil der wahren Spikes, die als Ereignis gemeldet wurden; darunter der Anteil der Ereignisse, die zu einem wahren Spike gehören.")
m3.metric("Sortiergenauigkeit", f"{dg.accuracy:.0%}", delta=f"Pipeline {pipe.accuracy:.0%} · Abgleich {tm_res.accuracy:.0%}", delta_color="off", help="Anteil der gefundenen wahren Spikes, die der richtigen Zelle zugeordnet sind.")
m4.metric("Nachbarschaften fehlend / falsch", f"{analysis.eta_m:.0%} / {analysis.eta_f:.0%}", delta=f"{_pct(analysis.single_contact)} der Zellen mit einem Kontakt", delta_color="off",
          help="Die Fehlermaße der Dissertation (η_m, η_f) für die Nachbarschaftsschätzung; darunter der Anteil der Zellen mit nur einem Kontakt.")

_t = vd
_others = f"Pipeline {_t['pipe_f1']:.2f}, Vorlagenabgleich {_t['tm_f1']:.2f}, ICA {_t.get('ica_f1', float('nan')):.2f}" + (f", SCA {_t['sca_f1']:.2f}" if "sca_f1" in _t else "")
if code == "delay_wins":
    st.success(f"✅ Der Verzögerungsgraph gewinnt: Spitzen-F1 {_t['dg']:.2f} gegen {_others}. Er erkennt {_t['recall']:.0%} der Spikes und ordnet {_t['accuracy']:.0%} davon richtig zu, ohne die Wellenform je zu sehen. "
               "Bedenken: das Szenario ist das Modell der Dissertation, für das der Verzögerungsgraph gebaut wurde - ändern Sie Jitter, Feuerrate oder p, um zu sehen, wo das kippt.")
elif code == "comparable":
    st.success(f"✅ Gleichauf: Spitzen-F1 des Verzögerungsgraphen {_t['dg']:.2f} gegen {_others}.")
elif code == "waveform_wins":
    st.warning(f"⚠️ Ein Wellenform-Verfahren ist hier besser: Spitzen-F1 {_t['dg']:.2f} des Verzögerungsgraphen gegen {_others}. Trefferquote {_t['recall']:.0%}, Genauigkeit der Detektion {_t['precision']:.0%}, "
               f"Sortiergenauigkeit {_t['accuracy']:.0%}.")
elif code == "jitter":
    st.warning(f"⚠️ Die Verzögerungen wackeln (Jitter {_t['jitter']:g} Abtastwerte): die feste Relativverzögerung, auf der der Verzögerungsgraph beruht, gibt es nicht mehr - er fällt auf {_t['dg']:.2f} (ohne Jitter {_t['dg'] + _t['drop']:.2f}). "
               f"Die Wellenform-Verfahren stört das kaum: {_others}.")
elif code == "rate":
    st.warning(f"⚠️ Manche Zellen feuern seltener als die angenommene Untergrenze von 5 Hz (Faktor {_t['rate_scale']:g}): die Kanten brauchen ν Wiederholungen, die Zelle liefert zu wenige - Spitzen-F1 {_t['dg']:.2f} (bei Faktor 1: {_t['dg'] + _t['drop']:.2f}). "
               f"Genauigkeit der Detektion {_t['precision']:.0%}, Nachbarschaften: {_t['eta_m']:.0%} fehlend, {_t['eta_f']:.0%} falsch. {_others}.")
elif code == "noise":
    st.warning(f"⚠️ Das Rauschen ({_t['noise']:g} µV) verwischt die Spitzenzeiten: Spitzen-F1 {_t['dg']:.2f} (bei 10 µV: {_t['dg'] + _t['drop']:.2f}); {_others}.")
elif code == "singles_excluded":
    st.warning(f"⚠️ Zellen mit nur einem Kontakt ({_t['single_contact']:.0%} der Zellen) sind ausgeschlossen: der Verzögerungsgraph erkennt ihre Spikes nicht (Trefferquote {_t['recall']:.0%}), was er meldet, stimmt aber (Sortiergenauigkeit {_t['accuracy']:.0%}). "
               f"Spitzen-F1 {_t['dg']:.2f} gegen {_others}. Mit der Einstellung \"zulassen\" gilt eine Spitze ohne bestätigte Kante als Zelle mit dieser Elektrode.")
elif code == "ambiguous":
    st.warning(f"⚠️ {_t['ambiguous_pairs']} Zellenpaare haben dieselbe normierte Charakteristik (dieselben Elektroden und relativen Verzögerungen, bei Einzelkontakt-Zellen: dieselbe Elektrode) - Proposition 2.3.1 der Dissertation: "
               f"sie sind aus den Spitzenzeiten nicht zu trennen und werden zu einer Zelle. Spitzen-F1 {_t['dg']:.2f} gegen {_others}.")

t1, t2 = st.columns(2)
with t1:
    st.markdown("**Raster: wahre Spikes (Striche) und ihre Zuordnung durch den Verzögerungsgraphen (Punkte)**")
    st.plotly_chart(build_raster(truth_times, truth_cell, sorting.times, dg.neuron_of_event, window, WINDOW_WIDTH_MS, m_n), width="stretch", key="res_raster")
with t2:
    st.markdown("**Spitzen-F1: Verzögerungsgraph, Pipeline, Vorlagenabgleich, ICA und SCA**")
    st.plotly_chart(build_method_bars(analysis), width="stretch", key="res_bars")
secs = analysis.seconds
st.caption(f"Punkte in der Zeile einer Zelle, ohne dass darüber ein Strich steht, sind falsch zugeordnet; Striche ohne Punkt darunter sind verpasste Spikes. Rechenzeit für diese Aufnahme: Verzögerungsgraph {secs['dg']:.2f} s, "
           f"Pipeline {secs['pipe']:.2f} s, Pipeline + Vorlagenabgleich {secs['matching']:.1f} s, ICA {secs.get('ica', float('nan')):.1f} s" + (f", SCA {secs['sca']:.0f} s" if "sca" in secs else "") + ".")

st.markdown("---")

# --- Sweeps ----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von den Eigenschaften der Aufnahme ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_OPTIONS), format_func=lambda p: SWEEP_OPTIONS[p], key="sweep_select")
current = {"n_cells": int(n_cells), "grid": int(grid), "contact_p": float(contact_p), "similarity": float(similarity), "delay_max": int(delay_max), "jitter": float(jitter), "rate_scale": float(rate_scale),
           "noise": float(noise), "seconds": float(seconds)}[sweep_param]
if st.button("Sweep über 5 feste Datensätze berechnen (dauert etwa eine Minute)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, data_key)}
if (sweep_param, data_key) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Datensätze..."):
        rows = _sweep(sweep_param, data_params[:9], settings)
    st.plotly_chart(build_sweep(rows, SWEEP_LABELS[sweep_param], current=current), width="stretch", key="sweep_chart")
    st.caption("Mittel über 5 feste Sweep-Datensätze (getrennt vom Seed oben), Band = Spanne des Verzögerungsgraphen; alle anderen Regler wie in der Seitenleiste. Rechts der Verzögerungsgraph im Detail. "
               "SCA fehlt in den Sweeps (nur bei mehr Zellen als Elektroden anwendbar und zu langsam).")

st.markdown("---")

# --- Szenen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🧩 Wer gewinnt wann: acht Szenen im Vergleich")
if st.button("Acht Szenen vergleichen (dauert etwa zwei Minuten)", key="scenes_start"):
    st.session_state["scenes_on"] = True
if st.session_state.get("scenes_on"):
    with st.spinner("Vergleiche 8 Szenen × 5 Datensätze × 4 Verfahren..."):
        scene_rows = _scenes(data_params[:9], settings)
    st.plotly_chart(build_scenes(scene_rows), width="stretch", key="scenes_chart")
    st.table({
        "Szene": [r["scene"] for r in scene_rows],
        "Verzögerungsgraph": [f"{r['dg']:.2f} ({r['dg_min']:.2f}-{r['dg_max']:.2f})" for r in scene_rows],
        "Pipeline": [f"{r['pipe']:.2f}" for r in scene_rows],
        "Abgleich": [f"{r['tm']:.2f}" for r in scene_rows],
        "ICA": [f"{r['ica']:.2f}" for r in scene_rows],
        "Zeit Verz.graph / Abgleich": [f"{r['sec_dg']:.1f} s / {r['sec_matching']:.1f} s" for r in scene_rows],
    })
    st.caption("Die ersten fünf Szenen sind Stärken des Verfahrens, die letzten drei Schwächen (Jitter, Feuerraten unter der Annahme, Einzelkontakt-Zellen auf wenigen Elektroden). Länge, Rauschen (sofern die Szene es nicht setzt) und die Einstellung für "
               "Einzelkontakt-Zellen wie in der Seitenleiste. Mittel und Spanne über 5 feste Datensätze; SCA gibt es hier nicht (siehe oben).")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden - und wer danach kommt")
st.markdown(
    """
| Annahme (Dissertation) | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Jede Zelle hat Kontakt zu mindestens zwei Elektroden** (Annahme 2.3.3) | Zellen mit einem Kontakt haben keine Verzögerung, die sie kennzeichnet: ausgeschlossen gehen ihre Spikes verloren (Preset "Einzelkontakt-Zellen ausgeschlossen"); zugelassen gilt "eine Spitze ohne Kante = Zelle dieser Elektrode", und mehrere Zellen einer Elektrode vermischen sich. | Die Wellenform-Verfahren (Pipeline, Vorlagenabgleich) - sie brauchen nur eine Elektrode |
| **Die Verzögerungen sind konstant** | Mit Jitter gibt es keine stabile Zeitdifferenz mehr: F1 fällt von 0.97 auf 0.70 bei Jitter 2 (Preset "Wackelnde Verzögerungen"). | Verfahren, die die Form einbeziehen; die Dissertation nennt die Kombination beider Ansätze als Ausblick |
| **Jede Zelle feuert mindestens ν Mal** (Annahme 2.3.1) | Seltene Zellen erreichen die Häufigkeitsschwellen nicht (Preset "Niedrige Feuerraten"); bei bekannter Untergrenze ist ν daran gekoppelt, bei zu hoher angenommener Untergrenze fehlen Zellen. | Längere Aufnahmen; Online-Verfahren, die Zellen nachträglich aufnehmen |
| **Spitzenzeiten auf wenige Abtastwerte genau** | Bei hohem Rauschen verschieben und verlieren sich Spitzen; die Nachbarschaftsfehler η steigen stark (20 µV: 20 % fehlend, 45 % falsch), die Zuordnung leidet weniger. | Bessere Spitzenerkennung, höhere Abtastrate (die Dissertation verdoppelt die Rate künstlich) |
| **Zellen unterscheiden sich in der normierten Charakteristik** (Prop. 2.3.1) | Gleiche Elektroden und gleiche relative Verzögerungen sind aus den Spitzenzeiten nicht trennbar - sie werden zu einer Zelle. | Die Wellenform (Vorlagenabgleich) trennt sie, sofern sich die Formen unterscheiden |
| **Greedy findet die richtige Clique** | Bei dichter Überlappung kann die Greedy-Clique falsch sein; die Dissertation lockert deshalb die Häufigkeitsschwelle auf ν/2. | Exakte Cliquensuche (die Dissertation zeigt sie für ideale Daten) |
"""
)
st.caption("Die Tabelle nennt Annahmen der Dissertation und was in dieser Demo bei ihrer Verletzung gemessen wird; ein Leistungsvergleich der Dissertation selbst mit anderen Verfahren gibt es dort nicht - dieser Vergleich ist neu und synthetisch.")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell (Abschn. 2.1 der Dissertation).** Elektroden $e_1,\dots,e_n$ auf einem Gitter; Zelle $z_i$ hat Kontakt zu einer Elektrodenmenge $N_i$ (Nachbarschaft) mit Verzögerungen $d_{ij}$. Feuert sie zur Zeit $s$, zeichnet $e_j$ ($j\in N_i$) die Vorlage
$\varphi_{ij}(t) = \alpha_1 e^{-(t-\beta_1)^2/2\gamma_1^2} + \alpha_2 e^{-(t-\beta_2)^2/2\gamma_2^2}$ ($\alpha_1<0<\alpha_2$, $\gamma_1\ll\gamma_2$) mit der Spitze bei $s + d_{ij}$ auf; dazu weißes Rauschen und kleine Signale weiter entfernter Zellen.

**Verzögerungsgraph.** Knoten: Spitzen $\pi_{j,l}$ der Elektroden. Kante zwischen $(j_1,l_1)$ und $(j_2,l_2)$, wenn die Zeitdifferenz $\pi_{j_2,l_2}-\pi_{j_1,l_1}$ **zulässig** ist: in einem Intervall der Länge $4\varepsilon$ liegen mindestens $(1-\theta_1)\nu$ Differenzen
aller Paare der beiden Elektroden mit Betrag $\le \delta = d_{\max}+2\varepsilon$. Eine Instanz (Spike) $I_{i,k}=\{(j,l): \pi_{j,l}=s_{i,k}+d_{ij}\}$ ist eine vollständige Teilmenge (Prop. 2.2.2).

**Nachbarschaft.** $\tilde N(j,l)=\{j\}\cup\{j' : (j,l)\text{ hat eine Kante zu }(j',\cdot)\}$; bestätigt, wenn die Menge mindestens $(1-\theta_2)\,\nu\,|N|$ Mal vorkommt. Fehler: $\eta_m=|\mathcal N\setminus\tilde{\mathcal N}|/|\mathcal N|$, $\eta_f=|\tilde{\mathcal N}\setminus\mathcal N|/|\mathcal N|$.

**Charakteristik und Zuordnung.** Normierte Charakteristik einer Clique $K$: $F_K(j')=\pi_{j',l'}-\pi_{j_{\min},l_{\min}}$ für $(j',l')\in K$ ($j_{\min}$ kleinste Elektrode). Kandidaten: Charakteristiken, die mindestens $\nu/2$ Mal vorkommen (hier mit Toleranz $2\varepsilon$ je Komponente gruppiert,
weil die Spitzen geschätzt sind). Jede Greedy-Clique (Algorithmus 5: iterativ den Knoten mit größtem Grad im Nachbarschaftsgraphen hinzunehmen) mit einem Kandidaten als Charakteristik ist ein Spike dieser Zelle.
**Proposition 2.3.1:** Zellen mit gleicher normierter Charakteristik sind aus den Spitzenzeiten nicht unterscheidbar.

**Bewertung.** Ereignisse (Zeit der Spitze auf der Elektrode mit kleinstem Index) und wahre Spikes (Zeit der ersten Spitze) werden zugeordnet, wenn ihre Zeiten höchstens 12 Abtastwerte auseinanderliegen; Zelle $\leftrightarrow$ Neuron: optimale Zuordnung (Ungarische Methode).
**Spitzen-F1** je Zelle aus den Zeiten ihres Neurons (Toleranz 12 Abtastwerte), gemittelt. Für ICA und SCA: Schätzspur je Zelle per Korrelation zugeordnet, Spitzen darauf erkannt.

Implementiert in `dg_delay.py` (Spitzen, Graph, Nachbarschaften, Cliquen, Charakteristiken), `dg_algorithm.py` und `dg_matching.py` (Pipeline und Vorlagenabgleich der Vorgänger), `dg_ica.py`, `dg_sca.py` (Vergleichsverfahren), `dg_scenario.py`
(Modell der Dissertation), `dg_evaluation.py` (Zuordnung, Kennzahlen, Sweeps, Szenen, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
