# Verzögerungsgraph – Zellen an ihren Zeitverzögerungen erkennen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-delay-graph-demo.streamlit.app/)**

Sechstes Stück der **Quellentrennung-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning": anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich)
zeigt diese Demo **ein** Verfahren – den **Verzögerungsgraph** aus der Dissertation des Autors (Universität Rostock, 2017) – an einem wachsenden Beispiel, mit **Standardpipeline**, **Vorlagenabgleich**, **ICA** und **SCA** als Vergleich.
Die bisherigen Spike-Sorting-Stücke ([spike-sorting-demo](../spike-sorting-demo), [template-matching-demo](../template-matching-demo)) erkennen Zellen an der **Wellenform**; der Verzögerungsgraph nutzt stattdessen, dass das Signal einer Zelle
verschiedene Elektroden mit **festen zeitlichen Verzögerungen** erreicht: Spitzen verschiedener Elektroden werden verbunden, wenn ihr Zeitabstand sich immer wieder gleich zeigt; zusammenhängende Gruppen (Cliquen) sind Spikes derselben Zelle,
die Menge ihrer Elektroden ist die Nachbarschaft der Zelle. Die Wellenform wird nie benutzt.

## Eigener Nachbau, Vergleich neu – kein Ergebnis der Dissertation

- Das Verfahren ist hier **von Grund auf nach der Dissertation nachgebaut** (Kap. 2: Spitzenschätzung, Verzögerungsgraph, Nachbarschaften, Greedy-Cliquen, Charakteristiken). Abweichungen stehen in `dg_delay.py` und weiter unten.
- Die **Dissertation vergleicht das Verfahren nicht mit Wellenform-Verfahren oder ICA**; sie bewertet es an eigenen Simulationen mit eigenen Fehlermaßen und nennt die Kombination beider Ansätze als Ausblick. Der Vergleich in dieser Demo ist **neu, synthetisch**
  und läuft auf **dem Modell der Dissertation** (Elektrodengitter, Kontakte, Verzögerungen, Zwei-Gauß-Vorlage, 40 kHz) – also dem Terrain, für das das Verfahren gebaut wurde. Siege dort sind keine Überraschung; deshalb gehören Szenen und Presets dazu,
  in denen es verliert.
- Das Spike-Sorting definiert die Dissertation nur für exakte Spitzenzeiten und ideale Daten; hier arbeitet es an geschätzten Spitzen (Charakteristiken mit Toleranz gruppiert, siehe unten). Eine Bewertung der Dissertation selbst ist das nicht.

```
ica-demo → sobi-demo → sca-demo
ica-demo → nmf-demo (Nichtnegativität statt Unabhängigkeit)
pca-demo + Clustering-Linie → spike-sorting-demo (Standardpipeline)
                              → template-matching-demo (Vorlagenabgleich: löst Überlappung auf)
                              → delay-graph-demo (Verzögerungsgraph: Zeitverzögerungen statt Wellenform)
```

| Frage | Ergebnis (12 Zellen, 5×5 Elektroden, Rauschen 10 µV, 3 s; Mittel über 5 feste Datensätze, Seeds 100000–100004; Spitzen-F1 der Zellen; Verzögerungsgraph / Pipeline / Vorlagenabgleich / ICA) |
|---|---|
| Standardfall | ✅ **0.97** / 0.73 / 0.85 / 0.72. Trefferquote 99 %, Sortiergenauigkeit 98 %, Nachbarschaften: 2 % fehlend, 3 % falsch. Der Verzögerungsgraph ist mehr als fünfmal schneller als Pipeline plus Abgleich |
| Viele Kontakte, gleiche Formen (20 Zellen, 4×4, p 0.5, Ähnlichkeit 0) | ✅ 0.95 / 0.55 / 0.85 / 0.43 (SCA 0.20) |
| Mehr Zellen als Elektroden (30 auf 25) | ✅ 0.94 / 0.47 / 0.74 / 0.50 (SCA 0.07); mit 6 Zellen 0.99 / 0.93 / 0.95 / 0.85 |
| Hohe Feuerrate (×4) | ✅ 0.96 / 0.48 / 0.82 / 0.74 (die Pipeline bricht bei Überlappung ein) |
| Rauschen | ✅ 20 µV: 0.88 / 0.58 / 0.69 / 0.57; 30 µV: 0.72 / 0.27 / 0.36 / 0.42; 40 µV: 0.60 / 0.10 / 0.11 / 0.22 – die Fehlermaße der Dissertation steigen dabei stark (20 µV: 20 % fehlende, 45 % falsche Nachbarschaften) |
| Wellenform-Ähnlichkeit | ✅ der Verzögerungsgraph sieht die Form nie: 0.99 bei Ähnlichkeit 0 (Pipeline 0.67, Abgleich 0.81) |
| Kürzere Aufnahme | ✅ 1 s genügt: 0.99 (Pipeline 0.72, Abgleich 0.73) |
| **Wackelnde Verzögerungen** | ❌ Jitter 1 / 2 / 3 Abtastwerte: **0.91 / 0.70 / 0.62**; der Abgleich hält 0.80 / **0.77** / 0.63 – bei Jitter 2 gewinnt er |
| **Niedrige Feuerraten** | ❌ Faktor 0.25 (unter der angenommenen Untergrenze 5 Hz): **0.72** gegen Pipeline **0.86** und Abgleich 0.82; Genauigkeit der Detektion 0.62, Nachbarschaften 57 % fehlend / 48 % falsch |
| **Kleines Gitter** | ❌ 3×3 Elektroden: 0.86 gegen Abgleich **0.92** (Pipeline 0.76, ICA 0.42): mehrere Zellen teilen sich dieselben Elektroden |
| **Einzelkontakt-Zellen** | p = 0.1 (63 % der Zellen mit einem Kontakt): 0.86 / 0.71 / 0.74 / 0.71; **ausgeschlossen** (nur Spikes mit bestätigter Kante) nur **0.37** (Trefferquote 0.36, aber alles Gemeldete stimmt: Sortiergenauigkeit 0.98); im Standardfall 0.77 statt 0.97 |
| **Nicht trennbare Zellen** | ❌ auf 3×3 mit 20 Zellen und p 0.1 gibt es im Mittel 8.8 Zellenpaare mit identischer Charakteristik (Prop. 2.3.1): 0.59 gegen Abgleich 0.68 (siehe Szenen) |
| Verzögerungsspanne | ⚠️ überraschend: der Verzögerungsgraph erreicht 0.99 auch **ohne** Verzögerungen (Spanne 0) und mit 8 – die **Elektrodenmengen** unterscheiden die Zellen schon; die Verzögerungen tragen in diesem Modell wenig. ICA leidet (0.77 → 0.62), weil die Mischung nicht augenblicklich ist |

## Was die Demo zeigt

1. **Verzögerungsgraph in Aktion** (Schritt-Slider + Abspielen, Zeitfenster): **Aufnahme** (Gitter mit Zellen und wahren Kontakten, Elektrodenspuren) → **Spitzen schätzen** (Signal, Glättung, Differenzfolge mit den Schwellen der Dissertation) →
   **Verzögerungsgraph** (Histogramm der Zeitdifferenzen eines Paares mit gemeinsamer Zelle und eines ohne; der Graph im Zeitfenster, Farbe = Zelle) → **Nachbarschaften** (Gitter: richtig / falsch / fehlend, η_m und η_f wie in der Dissertation) →
   **Zellen zuordnen** (Raster, Verwechslungsmatrix).
2. **Ergebnis im Vergleich:** Spitzen-F1 (für alle Verfahren dieselbe Definition), Trefferquote, Sortiergenauigkeit, Nachbarschaftsfehler, Rechenzeiten; Urteil (Codes: Verzögerungsgraph gewinnt / gleichauf / Wellenform gewinnt / Jitter / zu niedrige Feuerrate / Rauschen /
   Einzelkontakt-Zellen ausgeschlossen / nicht unterscheidbare Zellen – die Ursachen per Referenzlauf ohne die gestörte Eigenschaft).
3. **📐 Sweeps** über Zellen, Gitter, Kontaktwahrscheinlichkeit, Ähnlichkeit, Verzögerungsspanne, Jitter, Feuerrate, Rauschen und Länge (fünf feste Datensätze ab 100000, Band = Spanne; Experiment auf Abruf).
4. **🧩 Wer gewinnt wann:** acht Szenen – fünf Stärken, drei Schwächen – für Verzögerungsgraph, Pipeline, Abgleich und ICA mit Spanne und Rechenzeiten (Experiment auf Abruf).
5. **🚧 Grenzen:** Tabelle "Annahme der Dissertation – was passiert – wer setzt an".

Regler: Zellen (4–30), Gitter (3–6), Kontaktwahrscheinlichkeit, Wellenform-Ähnlichkeit, Verzögerungsspanne (0–8), Verzögerungs-Jitter (0–3), Feuerrate (×0.25–×4), Rauschen (2–40 µV), Länge (1–8 s), Zellen mit nur einem Kontakt (zulassen / ausschließen).

## Messwerte der Presets (Seed 7; sie prüfen sich mit weiten Bändern selbst)

| Preset | Verzögerungsgraph | Pipeline | Abgleich | ICA | Urteil |
|---|---|---|---|---|---|
| Standardfall (Dissertations-Modell) | 0.99 | 0.75 | 0.88 | 0.58 | Verzögerungsgraph gewinnt |
| Viele Kontakte, gleiche Formen | 0.96 | 0.54 | 0.90 | 0.48 (SCA 0.12) | gleichauf (Abstand unter der Marge von 0.08) |
| Mehr Zellen als Elektroden | 0.85 | 0.43 | 0.65 | 0.48 (SCA 0.07) | Verzögerungsgraph gewinnt |
| Wackelnde Verzögerungen (Jitter 2) | 0.79 | 0.62 | 0.86 | 0.56 | Jitter |
| Niedrige Feuerraten (×0.25) | 0.82 | 0.85 | 0.89 | 0.52 | gleichauf |
| Einzelkontakt-Zellen ausgeschlossen (p 0.1) | 0.33 | 0.76 | 0.77 | 0.70 | Einzelkontakt-Zellen ausgeschlossen |

## Modell und Verfahren

- **Szenario** (`dg_scenario.py`, nach Abschn. 2.1): Elektrodengitter (Zentren 0.5 + j), Zellen an Zufallsorten mit Kontakt zu Elektroden im Radius R = 2 mit Wahrscheinlichkeit p (Zellen ohne Kontakt werden ersetzt), Verzögerung je Kontakt gleichverteilt in [0, d_max],
  Vorlage je Kontakt aus zwei Gauß-Funktionen (Parameterintervalle wie in der Dissertation, 40 kHz, Breite 80), Erneuerungsprozess mit 1 ms Refraktärzeit und 5–30 Hz, weißes Rauschen (Standard 10 µV) und der Hintergrund pseudo-adjazenter Zellen (κ = 0.05).
  Neu als Regler: Wellenform-Ähnlichkeit, Verzögerungs-Jitter, Feuerraten-Faktor. Jede Zelle hat eigene Zufallsströme.
- **Verzögerungsgraph** (`dg_delay.py`, numpy von Grund auf): **Spitzenschätzung** je Elektrode (Binomialfilter, Differenz, Rechteckfilter, Schwellen, Abschn. 2.2.2); **Graph** (Algorithmus 1: Differenzen bis δ = d_max + 2ε; Algorithmus 2: zulässige Differenzen in einem Fenster der Länge 4ε
  mit mindestens (1 − θ₁)ν Vorkommen, θ₁ = 0.1); **Nachbarschaften** (Mengen mit mindestens (1 − θ₂)ν|N| Vorkommen, θ₂ = 0.2); **Greedy-Clique** (Algorithmus 5) und **Charakteristiken** (Algorithmus 3, modifiziert: Kandidaten mit mindestens ν/2 Vorkommen).
  ν = Aufnahmedauer × 5 Hz (untere Feuerrate, wie in der Dissertation als bekannt vorausgesetzt), d_max = die eingestellte Verzögerungsspanne (ebenfalls bekannt).
  **Abweichungen dieser Demo:** (1) mit mehr Rauschen als 10 µV werden die Schwellen entsprechend angehoben, bei weniger bleiben sie (sonst wird der 5-%-Hintergrund als Spike erkannt); (2) die Charakteristiken der Cliquen werden mit Toleranz 2ε je Komponente gruppiert, weil die Spitzen geschätzt
  statt exakt sind; (3) Cliquen, die in einer größeren enthalten sind, zählen nicht als eigener Spike; (4) Zellen mit nur einem Kontakt (Cliquen aus einem Knoten) sind wie in der Heuristik der Dissertation zugelassen, in der Oberfläche als Schalter.
- **Vergleich:** Standardpipeline (`dg_algorithm.py`) und Vorlagenabgleich (`dg_matching.py`) aus den Vorgängern, Zeitkonstanten auf 40 kHz skaliert, Pipeline mit so vielen Hauptkomponenten wie Zellen (höchstens 15; drei wie in den Vorgängern reichen für ein Dutzend Zellen nicht: 0.62 statt 0.73),
  Vorlagenabgleich mit FFT-Passung und Verwerfen der Nachbarschaft verworfener Reste (beides Beschleunigungen ohne Ergebnisänderung im Standardfall); ICA (`dg_ica.py`) und SCA (`dg_sca.py`) wortgleich. Alle Wellenform-Verfahren bekommen die wahre Zellzahl.
- **Auswertung** (`dg_evaluation.py`): für Verzögerungsgraph, Pipeline und Abgleich dieselbe Auswertung von Ereignissen (Zeit, Neuron) gegen die wahren Spikes (Zeit der ersten Spitze, Toleranz ±12 Abtastwerte = 0.3 ms; Kollision = anderer Spike innerhalb ±48 Abtastwerte), Zuordnung Zelle ↔ Neuron mit der
  **Ungarischen Methode** (die Bitmasken-DP der Vorgänger wächst mit 2^Spalten und ist bei 30 Zellen unbrauchbar; gegen Brute-Force geprüft); Spitzen-F1 je Zelle; ICA/SCA: Schätzspur je Zelle per Korrelation zugeordnet, Spitzen darauf erkannt. Dazu die Fehlermaße der Dissertation
  (η_m, η_f für die Nachbarschaften, ζ für die Charakteristiken), der Anteil Einzelkontakt-Zellen und die Zahl nicht trennbarer Zellenpaare (identische normierte Charakteristik, Prop. 2.3.1).

## Was nicht funktioniert hat / Grenzen

- **Die Vermutung "der Verzögerungsgraph gewinnt, wo die Wellenformen gleich sind" stimmt nur halb.** Bei gleichen Formen (Ähnlichkeit 0) behalten die Wellenform-Verfahren ihre Fußabdrücke über die Elektroden (Pipeline 0.67, Abgleich 0.81); der Vorsprung des Verzögerungsgraphen kommt aus den
  **Elektrodenmengen** der Zellen, nicht aus den Verzögerungen: ohne Verzögerungen (Spanne 0) erreicht er 0.99 wie mit ihnen. Die Verzögerungen werden erst wichtig, wo sich Zellen dieselben Elektroden teilen – und dort scheitert das Verfahren an Prop. 2.3.1, wenn auch die relativen Verzögerungen gleich sind.
- **Einzelkontakt-Zellen sind die eigentliche Schwäche, und die Heuristik der Dissertation verdeckt sie.** Ohne sie (nur Spikes mit bestätigter Kante) fällt der Verzögerungsgraph im Standardfall von 0.97 auf 0.77 (Spitzen-F1), bei p = 0.1 auf 0.37. Zugelassen erkennt er sie allein an ihrer Elektrode – das funktioniert,
  solange auf dieser Elektrode nur eine Zelle sitzt; sitzen mehrere darauf, werden sie zu einer (0.59 auf 3×3 mit 20 Zellen). Das ist keine Verzögerungs-Erkennung mehr, sondern "Spitze ohne Partner".
- **Jitter, zu niedrige Feuerraten und kleines Gitter sind echte Verluste** (siehe Tabelle) – Annahmen der Dissertation (konstante Verzögerungen, bekannte Untergrenze der Feuerrate, Zellen mit wenigen gemeinsamen Elektroden), die hier verletzt werden. Der Vorlagenabgleich ist dort jeweils besser oder gleich.
- **Die SCA ist in diesem Szenario nicht sinnvoll einzusetzen und hier nur bei mehr Zellen als Elektroden dabei.** Mit den Einstellungen der sca-demo (Aktivitätsschwelle 4 × Rausch-Norm, für 2–8 Elektroden abgestimmt) findet sie bei 25 Elektroden keine aktiven Zeitpunkte (F1 unter 0.2); mit Schwelle 2 wären es bei einer Aufnahme etwa 0.56, aber mehrere Sekunden je Analyse – zu langsam für eine interaktive Demo. Bei mehr Zellen als Elektroden bleibt sie bei 0.07–0.20. Das ist ein Ergebnis für diese Einstellungen, kein Urteil über SCA allgemein; auch ICA und SCA nehmen augenblickliche Mischung an, die es hier nicht gibt.
- **Spitzenschätzung mit rauschrelativen Schwellen scheiterte:** die erste Fassung skalierte die Schwellen der Dissertation mit dem geschätzten Rauschen – bei 2 µV Rauschen wurde der 5-%-Hintergrund pseudo-adjazenter Zellen als Spike erkannt, der Graph verklebte (F1 0.07). Jetzt bleiben die absoluten Schwellen bei
  Rauschen ≤ 10 µV und werden nur darüber angehoben.
- **Die Fehlermaße der Dissertation und die Zuordnung gehen auseinander:** bei 20 µV fehlen 20 % und 45 % der Nachbarschaften sind falsch, aber die Spikes werden noch mit F1 0.88 zugeordnet – einzelne falsche Nachbarschaften stören die Cliquen weniger, als η vermuten lässt.
- **Der Vorlagenabgleich wurde bei Rauschen unter 2 µV sehr langsam** (19 s statt 1.4 s: Reste weit über der Schwelle, jede Prüfung eine eigene Runde); deshalb Rauschen ab 2 µV und das Verwerfen der Nachbarschaft eines verworfenen Rests.
- **Nicht gebaut:** die Kombination beider Ansätze (Ausblick der Dissertation; etwa der Verzögerungsgraph als Start-Vorlagen für den Abgleich), das Verdoppeln der Abtastrate, die Cliquen-Überdeckung für fehlende Spikes und die Bandbreiten-Heuristik (Abschn. 2.5, Kap. 3). Auch nicht: Drift, korreliertes Rauschen, reale Daten
  (die Dissertation hat sie untersucht; die Ergebnisse waren dort schlechter, weil zu wenige Neuronen Kontakt zu mehr als einer Elektrode hatten).
- **Synthetische Daten:** das Modell der Dissertation, feste Vorlagen je Kontakt, exakt konstante Verzögerungen (außer beim Jitter-Regler), weißes Gauß'sches Rauschen. Die Vergleichsverfahren sind Nachbauten aus den Vorgänger-Demos, nicht die Werkzeuge der Praxis.
- **Die Spitzenerkennung des Spitzen-F1 (Kopie aus ica-demo) wandte die Mindesttiefe erst ab 10 gefundenen Spitzen an:** bei kürzeren Spuren blieben Rauschspitzen über 4 σ_MAD als Falschtreffer stehen, obwohl dieselbe Spur mit mehr Spitzen sie verworfen hätte. Jetzt gilt die Regel (30 % der typischen Tiefe, typische Tiefe = Median der höchstens 10 tiefsten Spitzen) auch dort, dann mit dem Median der gefundenen Spitzen. Gemessen über die 5 festen Sweep-Datensätze mit den Standard-Einstellungen: in allen 9 Sweeps ändert sich keine Kennzahl, und die Tests mit den in dieser Datei genannten Zahlen laufen unverändert grün. Als Test hinterlegt (`tests/test_detect_spikes_depth.py`).

## Verifikation

- Modell mit Handinstanzen und Eigenschaften (Vorlagenminimum am Anker, Kontakte im Radius, Durchschnittsgrad ≈ p·π·R², Reproduzierbarkeit unabhängig von der Zellzahl, Jitter ändert Spitzenzeiten aber nicht Feuerzeiten, Refraktärzeit); Spitzenschätzung (geplante Spitzen, Fehler höchstens wenige Abtastwerte);
  **Algorithmus 1 und 2 mit Handinstanzen**; Graph, Nachbarschaft und Greedy-Clique an geplanten Zellen; **Proposition 2.3.1** als Konstruktionstest (zwei Zellen mit gleicher normierter Charakteristik ergeben einen Kandidaten, mit verschiedener zwei); Einzelknoten-Cliquen nur bei `min_clique = 1`; Nachbarschaftsfehler mit Handinstanz.
- **Ungarische Methode gegen Brute-Force** (rechteckig, Gewinn 0); FFT-Passung gegen die direkte Rechnung; Matching Pursuit mit Handinstanzen; PCA, k-means und Silhouette gegen scikit-learn.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Zellen, Gitter, Kontakte, Ähnlichkeit, Verzögerungsspanne, Jitter, Feuerrate, Rauschen, Länge, Einzelkontakt-Zellen, Presets, Szenen – Stärken und Schwächen; jeweils Mittel über die festen Sweep-Datensätze mit weiten Toleranzen);
  Verdict-Codes über fünf Datensätze; alle 6 Presets in Bändern; AppTest-Rauchtests (Default, jedes Preset, jeder Schritt auch mit fast keinen Spikes, Randgrößen, Experimente auf Abruf), Achsensperre und explizite Schlüssel aller Figuren.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🧩 Szenen, 🚧 Grenzen, Mathe |
| `dg_delay.py` | Spitzenschätzung, Verzögerungsgraph, Nachbarschaften, Greedy-Clique, Charakteristiken, `run_delay_graph` |
| `dg_scenario.py`, `dg_constants.py` | Modell der Dissertation (Gitter, Kontakte, Verzögerungen, Vorlagen); Konstanten, Presets |
| `dg_algorithm.py`, `dg_matching.py` | Standardpipeline und Vorlagenabgleich der Vorgänger (auf 40 kHz skaliert) |
| `dg_ica.py`, `dg_sca.py` | Vergleichsverfahren (wortgleich aus den Vorgängern) |
| `dg_evaluation.py` | Ungarische Zuordnung, gemeinsame Auswertung, Fehlermaße der Dissertation, Sweeps, Szenen, Urteil |
| `dg_presets.py`, `dg_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Verzögerungsgraph und Modell, Vergleichsverfahren, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Quellentrennung: von ICA bis Verzögerungsgraph](https://sebastianhanisch.net/konzepte-quellentrennung.html).
