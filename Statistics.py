import numpy as np
import open3d as o3d


def show_basic_statistics(mesh, filename, labels_dictionary):
    print("\nNumber of vertices:", len(np.asarray(mesh.vertices)))
    print("Number of faces:", len(np.asarray(mesh.triangles)))
    box_points = o3d.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(mesh).get_box_points()
    print(f"Axis-aligned bounding box vertices coordinates: {np.asarray(box_points)}")

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



