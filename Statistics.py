import numpy as np
import open3d as o3d
from Mesh_Reading import load_mesh, check_type
from utils import get_complete_classification
from Normalization import get_barycenter
import csv
import os
from math import sqrt


def show_basic_statistics(mesh, filename, labels_dictionary):
    print("\nNumber of vertices:", len(np.asarray(mesh.vertices)))
    print("Number of faces:", len(np.asarray(mesh.triangles)))
    box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_box_points()
    print(f"Axis-aligned bounding box vertices coordinates: \n{np.asarray(box_points)}")

    max_aabbox_elongation = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_max_extent()
    barycenter = get_barycenter(mesh)
    distance_from_origin = sqrt(barycenter[0] ** 2 + barycenter[1] ** 2 + barycenter[2] ** 2)

    print("Max elongation of axis-aligned bounding box:", max_aabbox_elongation)
    print("Barycenter coordinates:", barycenter)
    print("Distance from origin:", distance_from_origin)

    label_class = labels_dictionary.get(int(filename[1:]))
    print(f"Shape belongs to class: {label_class}")

    print("\n------------- Additional info -------------\n")

    print("Values of triangle normals:", np.asarray(mesh.triangle_normals))

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(f"edge_manifold:          {edge_manifold}")
    print(f"edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"vertex_manifold:        {vertex_manifold}")
    print(f"self_intersecting:      {self_intersecting}")
    print(f"watertight:             {watertight}")
    print(f"orientable:             {orientable}")


def save_statistics(db_path):
    fieldnames = ["filename", "class", "vert_num", "face_num", "face_type", "AABBox", 'barycenter', 'distance_from_origin', 'max_elongation']
    output = []
    labels_dictionary = get_complete_classification()
    for (root, dirs, files) in os.walk(db_path):
        for filename in files:
            if filename.endswith(".off") or filename.endswith(".ply"):
                filepath = root+'/'+filename
                mesh = load_mesh(filepath)
                face_type = check_type(mesh)
                label_class = labels_dictionary.get(int(filename[1:-4]))
                vert_num = len(np.asarray(mesh.vertices))
                face_num = len(np.asarray(mesh.triangles))
                box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_box_points()
                box_array = np.asarray(box_points)

                max_aabbox_elongation = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_max_extent()
                barycenter = get_barycenter(mesh)
                distance_from_origin = sqrt(barycenter[0]**2 + barycenter[1]**2 + barycenter[2]**2)

                aabbox = '('+str(box_array[0][0])+', '+str(box_array[0][1])+', '+str(box_array[0][2])+'), ('\
                         +str(box_array[1][0])+', '+str(box_array[1][1])+', '+str(box_array[1][2])+'), ('\
                         +str(box_array[2][0])+', '+str(box_array[2][1])+', '+str(box_array[2][2])+'), ('\
                         +str(box_array[3][0])+', '+str(box_array[3][1])+', '+str(box_array[3][2])+'), ('\
                         +str(box_array[4][0])+', '+str(box_array[4][1])+', '+str(box_array[4][2])+'), ('\
                         +str(box_array[5][0])+', '+str(box_array[5][1])+', '+str(box_array[5][2])+'), ('\
                         +str(box_array[6][0])+', '+str(box_array[6][1])+', '+str(box_array[6][2])+'), ('\
                         +str(box_array[7][0])+', '+str(box_array[7][1])+', '+str(box_array[7][2])+')'

                output.append({'filename': filename, 'class': label_class, 'vert_num': vert_num, 'face_num': face_num,
                               'face_type': face_type, 'AABBox': aabbox, 'barycenter': barycenter, 'distance_from_origin': distance_from_origin,
                               'max_elongation': max_aabbox_elongation})
            else:
                continue

    with open('mesh_stats.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)

