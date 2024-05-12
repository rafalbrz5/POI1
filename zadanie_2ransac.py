import numpy as np
from numpy.linalg import svd

def fit_plane_RANSAC(points, iterations=1000, threshold=0.01):
    best_plane = None
    best_inliers = None
    max_inliers = 0

    for _ in range(iterations):
        # Losowo wybierz trzy punkty
        sample_indices = np.random.choice(len(points), 3, replace=False)
        sample_points = points[sample_indices]

        # Dopasuj płaszczyznę do wybranych punktów
        A = np.column_stack((sample_points, np.ones(3)))
        _, _, V = svd(A)
        plane_params = V[-1, :]

        # Oblicz odległość od płaszczyzny dla wszystkich punktów
        distances = np.abs(np.dot(points, plane_params[:-1]) + plane_params[-1]) / np.linalg.norm(plane_params[:-1])

        # Licz wewnętrzne punkty
        inliers = points[distances < threshold]
        num_inliers = len(inliers)

        # Aktualizuj najlepszą płaszczyznę, jeśli znaleziono lepszy model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_plane = plane_params
            best_inliers = inliers

    return best_plane, best_inliers

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

def k_means(points, k, iterations=100):
    centroids = points[np.random.choice(len(points), k, replace=False)]

    for _ in range(iterations):
        distances = np.linalg.norm(points[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            centroids[i] = np.mean(points[labels == i], axis=0)

    return centroids, labels

# Wczytaj chmurę punktów z pliku xyz
plane_vertical_points = load_point_cloud("plane_vertical_points.xyz")
plane_horizontal_points = load_point_cloud("plane_horizontal_points.xyz")
cylinder_points = load_point_cloud("cylinder_points.xyz")

# Znajdź rozłączne chmury punktów za pomocą algorytmu k-średnich
k = 3 
centroids_v, labels_v = k_means(plane_vertical_points, k)
centroids_h, labels_h = k_means(plane_horizontal_points, k)
centroids_c, labels_c = k_means(cylinder_points, k)


# Dopasuj płaszczyzny do każdej chmury punktów
def print_orientation(plane_name, points, labels):
    for i in range(k):
        normal_vector = fit_plane_RANSAC(points[labels == i])[0][:-1]
        if abs(normal_vector[2]) > abs(normal_vector[0]) and abs(normal_vector[2]) > abs(normal_vector[1]):
            orientation = "Pozioma"
        else:
            orientation = "Pionowa"
        print(f"{plane_name} {i + 1}:")
        print(f"Chmura punktów {i + 1}:")
        print("Wektor normalny:", normal_vector)
        print("Orientacja płaszczyzny:", orientation)
        print()


print_orientation("Płaszczyzna pionowa", plane_vertical_points, labels_v)

print_orientation("Płaszczyzna pozioma", plane_horizontal_points, labels_h)

print_orientation("Płaszczyzna cylindryczna", cylinder_points, labels_c)
