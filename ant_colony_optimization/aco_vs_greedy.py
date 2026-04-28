import numpy as np
import matplotlib.pyplot as plt
import random

# ==========================================
# 1. GÉNÉRATION DES DONNÉES (Points GPS)
# ==========================================
np.random.seed(42)
num_points = 20  # Nombre de points de livraison pour le drone
# Génération de coordonnées aléatoires (x, y) entre 0 et 100
coords = np.random.rand(num_points, 2) * 100

# Calcul de la matrice des distances (d_ij)
def calc_distances(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist_matrix

distances = calc_distances(coords)

# ==========================================
# 2. MÉTHODE DE BASE : PLUS PROCHE VOISIN
# ==========================================
def nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    unvisited = list(range(1, n))
    current_node = 0
    route = [current_node]
    total_dist = 0
    
    while unvisited:
        # Trouver le voisin non visité le plus proche
        next_node = min(unvisited, key=lambda node: dist_matrix[current_node][node])
        total_dist += dist_matrix[current_node][next_node]
        route.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
        
    # Retour à la base
    total_dist += dist_matrix[current_node][route[0]]
    route.append(route[0])
    return route, total_dist

# ==========================================
# 3. MÉTHODE ACO (ANT COLONY OPTIMIZATION)
# ==========================================
def aco_tsp(dist_matrix, n_ants=20, n_iterations=100, alpha=1.0, beta=2.0, rho=0.1, Q=100):
    n = len(dist_matrix)
    # Initialisation des phéromones (tau_0 = 1)
    pheromone = np.ones((n, n))
    
    # Heuristique locale : eta = 1 / distance (on met 0 sur la diagonale)
    eta = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                eta[i][j] = 1.0 / dist_matrix[i][j]

    best_route = None
    best_dist = float('inf')

    for iteration in range(n_iterations):
        all_routes = []
        all_distances = []
        
        # Construction des solutions par chaque fourmi
        for ant in range(n_ants):
            current_node = random.randint(0, n - 1)
            route = [current_node]
            unvisited = list(range(n))
            unvisited.remove(current_node)
            dist_tour = 0
            
            while unvisited:
                # Calcul des probabilités de transition
                probs = []
                for node in unvisited:
                    p = (pheromone[current_node][node] ** alpha) * (eta[current_node][node] ** beta)
                    probs.append(p)
                
                probs = np.array(probs) / sum(probs)
                
                # Choix probabiliste du prochain nœud
                next_node = np.random.choice(unvisited, p=probs)
                dist_tour += dist_matrix[current_node][next_node]
                
                route.append(next_node)
                unvisited.remove(next_node)
                current_node = next_node
                
            # Retour à la base
            dist_tour += dist_matrix[current_node][route[0]]
            route.append(route[0])
            
            all_routes.append(route)
            all_distances.append(dist_tour)
            
            # Mise à jour de la meilleure solution globale
            if dist_tour < best_dist:
                best_dist = dist_tour
                best_route = route
                
        # Mise à jour des phéromones (Offline)
        # 1. Évaporation
        pheromone *= (1 - rho)
        # 2. Renforcement
        for route, dist in zip(all_routes, all_distances):
            for i in range(n):
                u = route[i]
                v = route[i+1]
                pheromone[u][v] += Q / dist
                pheromone[v][u] += Q / dist # Graphe non orienté

    return best_route, best_dist

# ==========================================
# 4. EXÉCUTION ET VISUALISATION
# ==========================================
# Exécution du Plus Proche Voisin
nn_route, nn_dist = nearest_neighbor(distances)

# Exécution de l'ACO (Note: beta passé à 5.0 comme demandé implicitly dans le prompt précédent)
aco_route, aco_dist = aco_tsp(distances, n_ants=20, n_iterations=100, alpha=1.0, beta=5.0, rho=0.1, Q=100)

print(f"Distance Plus Proche Voisin : {nn_dist:.2f} km")
print(f"Distance ACO : {aco_dist:.2f} km")

# Affichage des graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) # Légère augmentation de la taille pour aérer

def plot_route(ax, route, coords, title, color, dist):
    x = [coords[i][0] for i in route]
    y = [coords[i][1] for i in route]
    ax.plot(x, y, marker='o', linestyle='-', color=color, markersize=8)
    ax.plot(x[0], y[0], marker='s', color='black', markersize=12, label="Base") # Point de départ
    # Ajout de padding au titre pour éviter les chevauchements
    ax.set_title(f"{title}\nDistance totale : {dist:.2f} km", pad=15)
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Coordonnée X")
    ax.set_ylabel("Coordonnée Y")

plot_route(ax1, nn_route, coords, "Plus Proche Voisin (Glouton)", "red", nn_dist)
plot_route(ax2, aco_route, coords, "Algorithme ACO (Dr Stone)", "green", aco_dist)

plt.tight_layout()

# ============================================================
# --- MODIFICATION ICI POUR SAUVEGARDER L'IMAGE ---
# ============================================================
nom_fichier_image = "comparaison_itineraires_drones.png"
# bbox_inches='tight' permet de s'assurer que les légendes ou titres ne sont pas coupés
plt.savefig(nom_fichier_image, dpi=300, bbox_inches='tight')
print(f"\n[INFO] Le graphique de comparaison a été sauvegardé sous : {nom_fichier_image}")
# ============================================================

# Afficher tout de même à l'écran
plt.show()