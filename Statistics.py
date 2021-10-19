import numpy as np
import open3d as o3d
from Mesh_Reading import load_mesh, check_type
from utils import get_complete_classification
from Normalization import get_barycenter
import csv
import os
from math import sqrt


def statistics_to_save(mesh, filename):

    stats = {}

    labels_dictionary = get_complete_classification()

    file_name = filename[:filename.rfind('.')]
    shape_number = int(file_name[1:])
    face_num = len(np.asarray(mesh.triangles))
    vert_num = len(np.asarray(mesh.vertices))
    face_type = check_type(mesh)
    label_class = labels_dictionary.get(int(filename[1:-4]))
    box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_box_points()
    box_array = np.asarray(box_points)

    aabbox_str = '(' + str(box_array[0][0]) + ', ' + str(box_array[0][1]) + ', ' + str(box_array[0][2]) + '), (' \
             + str(box_array[1][0]) + ', ' + str(box_array[1][1]) + ', ' + str(box_array[1][2]) + '), (' \
             + str(box_array[2][0]) + ', ' + str(box_array[2][1]) + ', ' + str(box_array[2][2]) + '), (' \
             + str(box_array[3][0]) + ', ' + str(box_array[3][1]) + ', ' + str(box_array[3][2]) + '), (' \
             + str(box_array[4][0]) + ', ' + str(box_array[4][1]) + ', ' + str(box_array[4][2]) + '), (' \
             + str(box_array[5][0]) + ', ' + str(box_array[5][1]) + ', ' + str(box_array[5][2]) + '), (' \
             + str(box_array[6][0]) + ', ' + str(box_array[6][1]) + ', ' + str(box_array[6][2]) + '), (' \
             + str(box_array[7][0]) + ', ' + str(box_array[7][1]) + ', ' + str(box_array[7][2]) + ')'

    max_aabbox_elongation = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_max_extent()
    barycenter = get_barycenter(mesh)
    distance_from_origin = sqrt(barycenter[0] ** 2 + barycenter[1] ** 2 + barycenter[2] ** 2)
    is_watertight = mesh.is_watertight()

    stats['file_name'] = file_name
    stats['shape_number'] = shape_number
    stats['face_num'] = face_num
    stats['vert_num'] = vert_num
    stats['face_type'] = face_type
    stats['label_class'] = label_class
    stats['box_points_array'] = box_array
    stats['box_points_string'] = aabbox_str
    stats['max_abbox_elongation'] = max_aabbox_elongation
    stats['barycenter'] = barycenter
    stats['distance_from_origin'] = distance_from_origin
    stats['is_watertight'] = is_watertight

    return stats


def show_shape_statistics(mesh, filename):

    stats = statistics_to_save(mesh, filename)

    print("\nFile name:", stats['file_name'])
    print("Shape number:", stats['shape_number'])
    print("Number of faces:", stats['face_num'])
    print("Number of vertices:", stats['vert_num'])
    print("The shape is composed by triangles or quads?:", stats['face_type'])
    print("Shape belongs to class:", stats['label_class'])
    print("Axis-aligned bounding box vertices coordinates:\n", stats['box_points_array'])
    print("Max elongation of axis-aligned bounding box:", stats['max_abbox_elongation'])
    print("Barycenter coordinates:", stats['barycenter'])
    print("Distance from origin:", stats['distance_from_origin'])


    print("\n----------------- Additional info -----------------\n")

    # compute and de-compute the normals
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    print("Values of triangle normals:", np.asarray(mesh.triangle_normals))
    mesh.vertex_normals = o3d.utility.Vector3dVector([])
    mesh.triangle_normals = o3d.utility.Vector3dVector([])

    # other useful info
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    orientable = mesh.is_orientable()

    print(f"edge_manifold:          {edge_manifold}")
    print(f"edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"vertex_manifold:        {vertex_manifold}")
    print(f"self_intersecting:      {self_intersecting}")
    print(f"watertight:             {stats['is_watertight']}")
    print(f"orientable:             {orientable}")

    print(stats)


def save_statistics(db_path, db_name):

    output = []

    for (root, dirs, files) in os.walk(db_path):

        for filename in files:

            if filename.endswith(".off") or filename.endswith(".ply"):
                filepath = root+'/'+filename
                mesh = load_mesh(filepath)

                stats = statistics_to_save(mesh, filename)
                del stats['box_points_array']

                output.append(stats)

                print("Statistics for", filename, "has been saved.")

            else:
                continue

    fieldnames = [i for i in stats]

    filename = db_name + '.csv'

    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
