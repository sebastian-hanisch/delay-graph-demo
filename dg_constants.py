"""Defaults, Regler-Grenzen und feste Größen der Verzögerungsgraph-Demo. Das Szenario folgt dem Modell der Dissertation (Abschn. 2.1: Elektrodengitter, Kontakte, Verzögerungen, Zwei-Gauß-Vorlage, 40 kHz);
die Vergleichsverfahren (Pipeline, Vorlagenabgleich, ICA, SCA) stammen aus den Vorgänger-Demos, ihre Zeitkonstanten sind auf 40 kHz skaliert (Faktor 4 gegenüber 10 kHz)."""

# --- Szenario nach der Dissertation (fest) ---------------------------------------------------------------------------------------
SAMPLE_RATE = 40_000                       # Hz (fr = 40 kHz; eine Zeiteinheit = 0.025 ms)
WIDTH = 80                                 # Breite der Vorlage w in Zeiteinheiten (2 ms)
RADIUS = 2.0                               # Kontakte nur zu Elektroden im Abstand R
REFRACTORY = 40                            # Refraktärzeit rho = fr Zeiteinheiten (1 ms)
RATE_RANGE = (5.0, 30.0)                   # Feuerraten der Zellen in Hz (gleichverteilt), mal Regler-Faktor
DELAY_MAX_DEFAULT = 4                      # d_max = w / 20
BACKGROUND_ATTENUATION = 0.05              # kappa: Dämpfung der Signale pseudo-adjazenter Zellen
# Intervalle der Vorlagenparameter (Dissertation, Abschn. 2.2.3: durch Anpassung an reale Daten gewonnen)
ALPHA1_RANGE = (-200.0, -50.0)             # muV
ALPHA2_FACTOR = (0.3, 0.6)                 # alpha2 in [-0.3 alpha1, -0.6 alpha1]
BETA1_FRACTION = (0.4, 0.525)              # beta1 in [0.4 w, 0.525 w]; beta2 = w / 2
GAMMA1_FRACTION = (0.025, 0.05)
GAMMA2_FRACTION = (0.1, 0.2)
REFERENCE_TEMPLATE = dict(alpha1=-100.0, alpha2=40.0, beta1=0.4625, gamma1=0.0375, gamma2=0.15)   # Ähnlichkeit 0: alle Kontakte dieser Form (beta, gamma als Bruchteil von w)

# --- Regler ---------------------------------------------------------------------------------------------------------------------
DEFAULT_GRID = 5
GRID_MIN, GRID_MAX = 3, 6                  # Gitter GRID x GRID
DEFAULT_N_CELLS = 12
N_CELLS_MIN, N_CELLS_MAX = 4, 30
DEFAULT_CONTACT_P = 0.2
CONTACT_P_MIN, CONTACT_P_MAX = 0.05, 0.5
DEFAULT_SIMILARITY = 1.0
SIMILARITY_MIN, SIMILARITY_MAX = 0.0, 1.0  # 1 = Vorlagen wie in der Dissertation, 0 = alle Kontakte aller Zellen dieselbe Form
DEFAULT_DELAY_MAX = DELAY_MAX_DEFAULT
DELAY_MAX_MIN, DELAY_MAX_MAX = 0, 8        # Verzögerungsspanne in Zeiteinheiten (0 = keine Verzögerungen)
DEFAULT_JITTER = 0.0
JITTER_MIN, JITTER_MAX = 0.0, 3.0          # Standardabweichung der Verzögerungsabweichung je Spike in Zeiteinheiten (0 = konstante Verzögerungen wie in der Dissertation)
DEFAULT_RATE_SCALE = 1.0
RATE_SCALE_MIN, RATE_SCALE_MAX = 0.25, 4.0
DEFAULT_NOISE = 10.0
NOISE_MIN, NOISE_MAX = 2.0, 40.0           # muV, Standardabweichung des weißen Rauschens (unter 2 wird der Abgleich sehr langsam: Reste weit über der Schwelle)
DEFAULT_SECONDS = 3.0
SECONDS_MIN, SECONDS_MAX = 1.0, 8.0
DEFAULT_SEED = 7
SWEEP_SEEDS = (100000, 100001, 100002, 100003, 100004)            # feste Datensätze der Sweeps, getrennt vom Demo-Seed

# --- Verzögerungsgraph (Dissertation Abschn. 2.2-2.4; Parameter wie dort) -----------------------------------------------------------
LOWER_RATE = 5.0                           # Hz: untere Schranke der Feuerrate -> nu = Länge * LOWER_RATE
THETA1 = 0.1                               # Fehlertoleranz Kanten
THETA2 = 0.2                               # Fehlertoleranz Nachbarschaften
EPSILON = max(1, WIDTH // 80)              # Maximalfehler der Spitzenschätzung
FILTER_B = WIDTH // 10                     # Binomialfilter (2b + 1 Werte)
DIFF_D = WIDTH // 16                       # Abstand der Differenz r_t - r_(t-d)
ETA_MINUS_UV = -50.0 / 3.0                 # Schwellen der Differenzfolge bei Rauschen 10 muV (werden mit dem geschätzten Rauschen skaliert)
ETA_PLUS_UV = 50.0 / 4.0
NOISE_REFERENCE_UV = 10.0
CANDIDATE_FRACTION = 0.5                   # Charakteristik-Kandidaten mindestens nu / 2 Mal (Dissertation Abschn. 2.4)

# --- Pipeline / Abgleich (aus den Vorgängern, auf 40 kHz skaliert) ------------------------------------------------------------------
SNIPPET_BEFORE = 24                        # Abtastwerte vor dem Minimum
SNIPPET_AFTER = 56                         # ... und ab dem Minimum (Länge 80 = Vorlagenbreite)
DEAD_TIME = 60                             # Abtastwerte Mindestabstand zweier erkannter Spitzen (1.5 ms)
MATCH_TOLERANCE = 12                       # erkannte und wahre Spitze gelten als dieselbe, wenn ihre Zeiten höchstens so weit auseinanderliegen (0.3 ms; die Verzögerungen betragen bis 8)
COLLISION_WINDOW = 48                      # ein wahrer Spike ist "Kollision", wenn ein anderer Spike höchstens so viele Abtastwerte entfernt beginnt (1.2 ms)
DETECT_TOLERANCE = 12                      # Spitzen-F1: erlaubte Abweichung zur wahren Spitze
DETECT_MIN_SEPARATION = 60
DEFAULT_THRESHOLD = 4.5
DEFAULT_FEATURE = "pca"
DEFAULT_N_COMPONENTS = 3                   # Vorgänger-Wert (4 Neuronen); hier wird die Komponentenzahl an die Zellzahl angepasst (siehe PCA_COMPONENTS_MAX)
PCA_COMPONENTS_MAX = 15                    # Komponenten der Pipeline = Zellzahl, höchstens so viele (gemessen: 3 Komponenten sind für 12 Zellen zu wenig, F1 0.62 gegen 0.73 mit 8)
DEFAULT_CLUSTER_MODE = "known"
K_MAX = 8
N_RESTARTS = 10
KMEANS_MAX_ITER = 100
DEFAULT_MATCH_THRESHOLD = 5.0
DEFAULT_MIN_AMPLITUDE = 0.5
DEFAULT_REFINE = "recluster"
DEFAULT_ROUNDS = 1

# --- Vergleichsverfahren ICA / SCA (aus den Vorgängern) -----------------------------------------------------------------------------
CONTRASTS = ("logcosh", "exp", "cube")
DEFAULT_CONTRAST = "logcosh"
METHODS = ("symmetric", "deflation")
DEFAULT_METHOD = "symmetric"
DEFAULT_INIT_START = 1
RECONSTRUCTIONS = ("l1", "single")
DEFAULT_RECONSTRUCTION = "l1"
MAX_ITER = 200                             # FastICA
TOL = 1e-6


# --- Verzögerungsgraph: Einzelkontakt-Zellen ----------------------------------------------------------------------------------------------
SINGLE_MODES = ("allowed", "excluded")
SINGLE_LABELS = {"allowed": "zulassen (Heuristik der Dissertation)", "excluded": "ausschließen (nur Spikes mit bestätigter Kante)"}
DEFAULT_SINGLES = "allowed"
SINGLE_CLIQUE = {"allowed": 1, "excluded": 2}          # kleinste Cliquengröße


# --- Presets ---------------------------------------------------------------------------------------------------------------------


def _preset(**kw):
    base = dict(m=DEFAULT_N_CELLS, grid=DEFAULT_GRID, contact_p=DEFAULT_CONTACT_P, similarity=DEFAULT_SIMILARITY, delay_max=DEFAULT_DELAY_MAX, jitter=DEFAULT_JITTER, rate_scale=DEFAULT_RATE_SCALE,
                noise=DEFAULT_NOISE, seconds=DEFAULT_SECONDS, singles=DEFAULT_SINGLES, seed=DEFAULT_SEED)
    base.update(kw)
    return base


PRESETS = {
    "Standardfall (Dissertations-Modell)": _preset(),
    "Viele Kontakte, gleiche Formen": _preset(contact_p=0.5, similarity=0.0, m=20, grid=4),
    "Mehr Zellen als Elektroden": _preset(m=30),
    "Wackelnde Verzögerungen": _preset(jitter=2.0),
    "Niedrige Feuerraten": _preset(rate_scale=0.25),
    "Einzelkontakt-Zellen ausgeschlossen": _preset(contact_p=0.1, singles="excluded"),
}
PRESET_HELP = {
    "Standardfall (Dissertations-Modell)": "12 Zellen auf 5×5 Elektroden nach dem Modell der Dissertation: Spitzen-F1 des Verzögerungsgraphen 0.97 gegen 0.73 der Pipeline, 0.85 des Vorlagenabgleichs und 0.72 von ICA (Mittel über fünf Aufnahmen). "
                                           "Er trifft 99 % der Spikes und ordnet 98 % davon richtig zu, ohne die Wellenform je zu sehen - im Terrain, für das das Verfahren gebaut wurde.",
    "Viele Kontakte, gleiche Formen": "20 Zellen auf 4×4 Elektroden, jede Zelle mit vielen Kontakten (p = 0.5), alle Zellen mit derselben Form und Amplitude: die Zellen unterscheiden sich nur über Elektrodenmengen und Verzögerungen. "
                                      "Verzögerungsgraph 0.95, Abgleich 0.85, Pipeline 0.55, ICA 0.43, SCA 0.20.",
    "Mehr Zellen als Elektroden": "30 Zellen auf 25 Elektroden: der Verzögerungsgraph bleibt bei 0.94 (Abgleich 0.74, ICA 0.50, Pipeline 0.47), ICA kann nur 25 Quellen finden. SCA, die für mehr Zellen als Elektroden gedacht ist, erreicht mit den Einstellungen "
                                  "der sca-demo (für 2-8 Elektroden abgestimmt) nur 0.07. Im Mittel gibt es in diesen Aufnahmen ein Zellenpaar mit identischer normierter Charakteristik (Prop. 2.3.1).",
    "Wackelnde Verzögerungen": "Die Verzögerung jeder Spitze weicht zufällig um 2 Abtastwerte ab: die feste Relativverzögerung, auf der das Verfahren beruht, gibt es nicht mehr. Spitzen-F1 0.70 statt 0.97; der Vorlagenabgleich hält 0.77, "
                               "die Pipeline 0.64, ICA 0.62 - hier gewinnt der Abgleich.",
    "Niedrige Feuerraten": "Feuerraten ×0.25 (1.25-7.5 Hz): unter der angenommenen Mindestrate von 5 Hz, aus der die Häufigkeitsschwellen abgeleitet sind. Spitzen-F1 0.72 gegen 0.86 der Pipeline und 0.82 des Abgleichs; die Genauigkeit der Detektion fällt auf 0.62 und die Nachbarschaften werden "
                           "unzuverlässig (57 % fehlend, 48 % falsch). In dieser Aufnahme (Seed 7) ist es näher: 0.82 gegen 0.89.",
    "Einzelkontakt-Zellen ausgeschlossen": "p = 0.1: 63 % der Zellen berühren nur eine Elektrode; die Einstellung schließt sie aus (nur Spikes mit bestätigter Kante zählen). Spitzen-F1 0.37 - was gemeldet wird, stimmt (Sortiergenauigkeit 0.98), aber nur 36 % der Spikes werden "
                                           "gefunden -, gegen 0.71 der Pipeline, 0.74 des Abgleichs und 0.71 von ICA. Mit \"zulassen\" (Heuristik der Dissertation) sind es 0.86.",
}
# Bänder (Seed des Presets; Werte mit dem ausgelieferten Code kalibriert, bewusst weit): Spitzen-F1 des Verzögerungsgraphen (f1), Trefferquote (recall), erlaubte Urteile (verdict)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Dissertations-Modell)": {"f1": (0.9, 1.0), "verdict": ("delay_wins", "comparable")},
    "Viele Kontakte, gleiche Formen": {"f1": (0.85, 1.0), "verdict": ("delay_wins", "comparable")},
    "Mehr Zellen als Elektroden": {"f1": (0.7, 1.0), "verdict": ("delay_wins",)},
    "Wackelnde Verzögerungen": {"f1": (0.6, 0.9), "verdict": ("jitter",)},
    "Niedrige Feuerraten": {"f1": (0.6, 0.95), "verdict": ("comparable", "waveform_wins", "rate")},
    "Einzelkontakt-Zellen ausgeschlossen": {"f1": (0.15, 0.55), "recall": (0.25, 0.55), "verdict": ("singles_excluded",)},
}
