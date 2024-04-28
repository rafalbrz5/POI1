import numpy as np
from scipy.stats import norm

def save_points_to_file(points, filename):
    with open(filename, 'w', encoding='utf-8', newline='\n') as csvfile:
        for p in points:
            csvfile.write(','.join(map(str, p)) + '\n')

def generate_points_plane_horizontal(width: float, length: float, height: float, filename: str, num_points: int = 2000):
    distribution_x = norm(loc=0, scale=width/3)
    distribution_y = norm(loc=0, scale=length/3)
    distribution_z = norm(loc=0, scale=height/3)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = np.column_stack((x, y, z))
    save_points_to_file(points, filename)

def generate_points_plane_vertical(width: float, height: float, length: float, filename: str, num_points: int = 2000):
    distribution_x = norm(loc=0, scale=width/3)
    distribution_y = norm(loc=0, scale=height/3)
    distribution_z = norm(loc=0, scale=length/3)

    x = distribution_x.rvs(size=num_points)
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = np.column_stack((x, y, z))
    save_points_to_file(points, filename)

def generate_points_cylinder(radius: float, height: float, filename: str, num_points: int = 2000):
    distribution_theta = norm(loc=0, scale=2*np.pi)
    distribution_z = norm(loc=0, scale=height/3)

    theta = distribution_theta.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    points = np.column_stack((x, y, z))
    save_points_to_file(points, filename)

# Przykłady użycia:
generate_points_plane_horizontal(width=50, length=100, height=10, filename='plane_horizontal_points.xyz')
generate_points_plane_vertical(width=30, height=40, length=80, filename='plane_vertical_points.xyz')
generate_points_cylinder(radius=15, height=50, filename='cylinder_points.xyz')
