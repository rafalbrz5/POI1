import numpy as np
import pyransac3d as pyrsc

def load_point_cloud(filename):
    points = []
    with open(filename, 'r') as file:
        for line in file:
            # Pomiń komentarze
            if line.startswith('#'):
                continue
            # Podziel linie na elementy, zakładając, że są oddzielone przecinkiem
            elements = line.strip().split(',')
            # Wczytaj współrzędne x, y, z
            x, y, z = map(float, elements[:3])
            points.append([x, y, z])
    return np.array(points)

def fit_plane_pyransac(points, thresh=0.01, min_points=3):
    plane = pyrsc.Plane()
    best_plane, best_inliers = plane.fit(points, thresh=thresh, minPoints=min_points)

    if best_plane is not None:
        # Pobierz współczynniki płaszczyzny
        normal_vector = best_plane[:-1]
        return normal_vector, best_inliers
    else:
        return None, None

def print_orientation_pyransac(plane_name, points):
    normal_vector, cluster_points = fit_plane_pyransac(points)
    if normal_vector is not None:
        if abs(normal_vector[2]) > abs(normal_vector[0]) and abs(normal_vector[2]) > abs(normal_vector[1]):
            orientation = "Pozioma"
        else:
            orientation = "Pionowa"
        print(f"{plane_name}:")
        print("Wektor normalny:", normal_vector)
        print("Orientacja płaszczyzny:", orientation)
        print()
    else:
        print(f"{plane_name}:")
        print("Nie udało się znaleźć płaszczyzny")
        print()

def separate_clusters_DBSCAN(points, eps=0.1, min_samples=50):
    plane = pyrsc.Plane()
    labels = plane.fit_predict(points, eps=eps, min_samples=min_samples)

    # Znajdź unikalne etykiety klastrów
    unique_labels = np.unique(labels)

    clusters = []
    for label in unique_labels:
        if label == -1:
            continue  # Pomijamy punkty uznane za szum
        cluster_points = points[labels == label]
        clusters.append(cluster_points)

    return clusters

# Wczytaj chmury punktów z plików xyz
plane_vertical_points_1 = load_point_cloud("plane_vertical_points.xyz")
plane_vertical_points_2 = load_point_cloud("plane_vertical_points.xyz")
plane_vertical_points_3 = load_point_cloud("plane_vertical_points.xyz")

plane_horizontal_points_1 = load_point_cloud("plane_horizontal_points.xyz")
plane_horizontal_points_2 = load_point_cloud("plane_horizontal_points.xyz")
plane_horizontal_points_3 = load_point_cloud("plane_horizontal_points.xyz")

cylinder_points_1 = load_point_cloud("cylinder_points.xyz")
cylinder_points_2 = load_point_cloud("cylinder_points.xyz")
cylinder_points_3 = load_point_cloud("cylinder_points.xyz")

print_orientation_pyransac("Płaszczyzna pionowa 1", plane_vertical_points_1)
print_orientation_pyransac("Płaszczyzna pionowa 2", plane_vertical_points_2)
print_orientation_pyransac("Płaszczyzna pionowa 3", plane_vertical_points_3)

print_orientation_pyransac("Płaszczyzna pozioma 1", plane_horizontal_points_1)
print_orientation_pyransac("Płaszczyzna pozioma 2", plane_horizontal_points_2)
print_orientation_pyransac("Płaszczyzna pozioma 3", plane_horizontal_points_3)

print_orientation_pyransac("Płaszczyzna cylindryczna 1", cylinder_points_1)
print_orientation_pyransac("Płaszczyzna cylindryczna 2", cylinder_points_2)
print_orientation_pyransac("Płaszczyzna cylindryczna 3", cylinder_points_3)
