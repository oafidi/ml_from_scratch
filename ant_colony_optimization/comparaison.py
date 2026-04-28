"""
Comparaison équitable de 4 approches ACO pour la sélection de features.
Corrections apportées :
  1. Corrélation réellement implémentée dans le mode Valadi
  2. Sélection BACO correcte (poids sélection vs non-sélection par feature)
  3. Mise à jour phéromone une seule fois par itération
  4. Sigmoid pour le DE (binarisation correcte)
  5. Seed fixe pour reproductibilité
  6. Courbes de convergence en plus du graphique final
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ─── Reproductibilité ────────────────────────────────────────
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# 1. DONNÉES + HEURISTIQUES PRÉ-CALCULÉES
# ══════════════════════════════════════════════════════════════
data = load_breast_cancer()
X_raw, y = data.data, data.target
X = StandardScaler().fit_transform(X_raw)   # normalisation
n_features = X.shape[1]

def get_fitness(indices):
    """Accuracy brute (CV-3) sans aucune pénalité."""
    if len(indices) == 0:
        return 0.0
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    # Suppression de la pénalité : on ne retourne que la moyenne
    return cross_val_score(clf, X[:, indices], y, cv=3).mean()


# Gini importance (filtre pré-calculé une seule fois)
rf_aux = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
gini_imp = rf_aux.feature_importances_
gini_norm = gini_imp / (gini_imp.sum() + 1e-9)

# Pénalité de corrélation (utilisée dans modes Valadi et BACO-DE-Gini)
corr_matrix = np.abs(np.corrcoef(X.T))
np.fill_diagonal(corr_matrix, 0)
corr_penalty = corr_matrix.mean(axis=1)
corr_penalty_norm = corr_penalty / (corr_penalty.max() + 1e-9)

# Heuristique hybride Gini × (1 - corrélation) — Valadi 2021
heuristic_valadi = np.clip(gini_norm * (1.0 - 0.5 * corr_penalty_norm), 0, 1)

# ══════════════════════════════════════════════════════════════
# 2. BACO BINAIRE CORRECT : décision par feature
# ══════════════════════════════════════════════════════════════

def baco_build_solution(pheromone, heuristic, q0=0.6):
    """
    Construction BACO correcte.
    Pour chaque feature i, deux poids :
      w_select     = pheromone[i,1] × heuristic[i]
      w_not_select = pheromone[i,0] × (1 - heuristic[i])
    Exploitation (prob q0)  : choisir le poids max
    Exploration  (prob 1-q0): tirage probabiliste
    """
    n = len(heuristic)
    subset = np.zeros(n, dtype=int)
    for i in range(n):
        w1 = pheromone[i, 1] * heuristic[i]
        w0 = pheromone[i, 0] * (1.0 - heuristic[i] + 1e-9)
        if np.random.rand() < q0:          # exploitation
            subset[i] = 1 if w1 > w0 else 0
        else:                               # exploration
            prob = w1 / (w1 + w0 + 1e-9)
            subset[i] = 1 if np.random.rand() < prob else 0
    return subset

# ══════════════════════════════════════════════════════════════
# 3. OPÉRATEUR DE MUTATION DE (correct)
# ══════════════════════════════════════════════════════════════

def de_mutate(population, F=0.8, CR=0.9):
    """
    Mutation DE adaptée aux vecteurs binaires.
    V = A + F*(B-C) → sigmoid → probabilité → binaire
    La sigmoid permet une binarisation douce (pas de seuil brutal).
    """
    pop_size, n = population.shape
    new_pop = np.zeros_like(population)
    for i in range(pop_size):
        cands = [j for j in range(pop_size) if j != i]
        a, b, c = population[np.random.choice(cands, 3, replace=False)]
        # Mutation continue
        v_cont = a.astype(float) + F * (b.astype(float) - c.astype(float))
        # Sigmoid → probabilité de sélection ∈ [0,1]
        v_prob = 1.0 / (1.0 + np.exp(-v_cont))
        v_bin  = (np.random.rand(n) < v_prob).astype(int)
        # Croisement
        mask = np.random.rand(n) < CR
        new_pop[i] = np.where(mask, v_bin, population[i])
    return new_pop

# ══════════════════════════════════════════════════════════════
# 4. MOTEUR DE RECHERCHE UNIFIÉ (corrigé)
# ══════════════════════════════════════════════════════════════

def run_experiment(mode, n_ants=12, n_iter=20, q0=0.6, evaporation=0.1):
    """
    Moteur unique, comportement contrôlé par 'mode'.

    Modes disponibles :
      "ACO"         → ACO seul, heuristique uniforme
      "Valadi"      → ACO + Gini + Corrélation (Valadi 2021), sans DE
      "Khushaba"    → ACO + DE, heuristique uniforme (Khushaba 2008)
      "BACO-DE-Gini"→ ACO + DE + Gini + Corrélation (notre approche)
    """
    # Choix de l'heuristique
    if "Gini" in mode or "Valadi" in mode:
        heuristic = heuristic_valadi      # Gini × (1 - corrélation)
    else:
        heuristic = np.ones(n_features) * 0.5   # uniforme

    # Phéromone : matrice (n_features × 2) → [i,0]=non-sél, [i,1]=sél
    pheromone = np.ones((n_features, 2)) * 0.5

    best_score   = -np.inf
    best_subset  = None
    history      = []    # courbe de convergence

    for it in range(n_iter):

        # ── Phase ACO : construction guidée ──────────────────
        aco_pop, aco_sc = [], []
        for _ in range(n_ants):
            sol = baco_build_solution(pheromone, heuristic, q0)
            sc  = get_fitness(np.where(sol == 1)[0])
            aco_pop.append(sol)
            aco_sc.append(sc)

        # ── Phase DE : recombinaison globale ─────────────────
        if "DE" in mode or "Khushaba" in mode:
            aco_array = np.array(aco_pop)
            de_pop    = de_mutate(aco_array)
            de_sc     = [get_fitness(np.where(s == 1)[0]) for s in de_pop]
            all_sol   = np.vstack([aco_array, de_pop])
            all_sc    = np.array(aco_sc + de_sc)
        else:
            all_sol = np.array(aco_pop)
            all_sc  = np.array(aco_sc)

        # ── Sélection de la meilleure solution ───────────────
        best_iter_idx = int(np.argmax(all_sc))
        best_iter_sc  = all_sc[best_iter_idx]
        best_iter_sol = all_sol[best_iter_idx]

        if best_iter_sc > best_score:
            best_score  = best_iter_sc
            best_subset = best_iter_sol.copy()

        # ── Mise à jour phéromone (une seule fois par iter) ──
        pheromone *= (1 - evaporation)          # évaporation globale
        for i in range(n_features):
            state = best_iter_sol[i]
            pheromone[i, state] += best_iter_sc  # récompense meilleure solution
        pheromone = np.clip(pheromone, 0.01, 10.0)

        history.append(best_score)

    n_selected = int(best_subset.sum()) if best_subset is not None else 0
    return best_score, n_selected, history

# ══════════════════════════════════════════════════════════════
# 5. LANCEMENT
# ══════════════════════════════════════════════════════════════

METHODS = [
    ("Khushaba 2008\n(ACO+DE)", "Khushaba"),
    ("Valadi 2021\n(ACO+Gini+Corr)", "Valadi"),
    ("BACO-DE-Gini\n(Notre approche)", "BACO-DE-Gini"),
]

print("Comparaison en cours...\n")
print(f"{'Méthode':<30} {'Score':>8} {'#Features':>10}")
print("-" * 52)

results = {}
for label, mode in METHODS:
    np.random.seed(42)   # seed identique pour chaque méthode
    score, n_sel, history = run_experiment(mode)
    results[label] = {"score": score, "n_sel": n_sel, "history": history}
    print(f"{label.replace(chr(10), ' '):<30} {score:>8.4f} {n_sel:>10}/{n_features}")

# ══════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════

COLORS = {
    "ACO Standard":                  "#7f8c8d",
    "Khushaba 2008\n(ACO+DE)":       "#2980b9",
    "Valadi 2021\n(ACO+Gini+Corr)":  "#e67e22",
    "BACO-DE-Gini\n(Notre approche)":"#27ae60",
}

fig = plt.figure(figsize=(14, 6))
gs  = gridspec.GridSpec(1, 2, width_ratios=[1, 1.3], wspace=0.35)

# ── Graphique 1 : Scores finaux ───────────────────────────────
ax1 = fig.add_subplot(gs[0])
labels = [k for k, _ in METHODS]
scores = [results[k]["score"] for k in labels]
colors = [COLORS[k] for k in labels]
short_labels = [k.split("\n")[0] for k in labels]

bars = ax1.bar(short_labels, scores, color=colors, edgecolor="white",
               linewidth=1.5, width=0.6)
ax1.set_ylim(min(scores) - 0.015, max(scores) + 0.012)
ax1.set_ylabel("Score (accuracy − pénalité)", fontsize=11)
ax1.set_title("Score final par méthode\n(Breast Cancer, 30 features)", fontsize=11)
ax1.tick_params(axis="x", labelsize=9)

for bar, sc, key in zip(bars, scores, labels):
    n = results[key]["n_sel"]
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.0015,
             f"{sc:.4f}\n({n} feat.)",
             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.spines[["top", "right"]].set_visible(False)

# ── Graphique 2 : Courbes de convergence ──────────────────────
ax2 = fig.add_subplot(gs[1])
for key, mode in METHODS:
    hist = results[key]["history"]
    ax2.plot(range(1, len(hist) + 1), hist,
             label=key.replace("\n", " "),
             color=COLORS[key],
             linewidth=2.2,
             marker="o", markersize=3)

ax2.set_xlabel("Itération", fontsize=11)
ax2.set_ylabel("Meilleur score cumulé", fontsize=11)
ax2.set_title("Courbes de convergence\n(meilleur score à chaque itération)", fontsize=11)
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(linestyle="--", alpha=0.5)
ax2.spines[["top", "right"]].set_visible(False)

plt.suptitle("Comparaison des approches ACO pour la sélection de features — Breast Cancer",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nGraphique sauvegardé.")
