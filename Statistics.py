def show_basic_statistics(mesh):
    print("Number of vertices:", len(mesh.vertices))
    print("Number of faces:", len(mesh.faces))
    print("Mass properties:", mesh.mass_properties)
    print("1:", mesh.bounding_box.extents)
    print("2:", mesh.bounding_box_oriented.primitive.extents)
    print("3:", mesh.bounding_box_oriented.primitive.transform)
