import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import random

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def load_scene(file_path):
    # Load the scene using trimesh
    scene = trimesh.load(file_path, force='scene')
    if isinstance(scene, trimesh.Scene):
        # If it's a scene, combine all geometries into a single mesh
        combined_mesh = trimesh.util.concatenate(scene.dump())
        return combined_mesh
    return scene

def compute_oriented_bounding_box(mesh):
    points = mesh.vertices
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Compute covariance matrix of the convex hull points
    cov_matrix = np.cov(hull_points, rowvar=False)
    
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues in descending order
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    
    # Transform points to the new coordinate system
    transformed_points = np.dot(hull_points, eigenvectors)
    
    # Compute min and max points in the new coordinate system
    min_point = np.min(transformed_points, axis=0)
    max_point = np.max(transformed_points, axis=0)
    
    # Compute the oriented bounding box corners in the new coordinate system
    obb_corners = np.array([
        [min_point[0], min_point[1], min_point[2]],
        [min_point[0], min_point[1], max_point[2]],
        [min_point[0], max_point[1], min_point[2]],
        [min_point[0], max_point[1], max_point[2]],
        [max_point[0], min_point[1], min_point[2]],
        [max_point[0], min_point[1], max_point[2]],
        [max_point[0], max_point[1], min_point[2]],
        [max_point[0], max_point[1], max_point[2]]
    ])
    
    # Transform OBB corners back to the original coordinate system
    obb_corners = np.dot(obb_corners, eigenvectors.T)
    
    return obb_corners

def compute_obb_lengths(obb_corners):
    # Compute lengths along each axis
    lengths = np.max(obb_corners, axis=0) - np.min(obb_corners, axis=0)
    return lengths

def get_obb_lengths(file_path):
    scene = load_scene(file_path)
    obb_dimensions = []

    if isinstance(scene, trimesh.Trimesh):
        obb_corners = compute_oriented_bounding_box(scene)
        obb_lengths = compute_obb_lengths(obb_corners)
        obb_dimensions.append(obb_lengths)
    else:
        for name, geom in scene.geometry.items():
            print(f"Processing geometry: {name}")
            obb_corners = compute_oriented_bounding_box(geom)
            obb_lengths = compute_obb_lengths(obb_corners)
            obb_dimensions.append(obb_lengths)

    return obb_dimensions