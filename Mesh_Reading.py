
import trimesh
import pyrender


def load_mesh(filepath):
    """
    :param filepath: the path of the 3D object. This function can handle both .ply
    and .off formats, as well as others.
    :return: a loaded mesh
    """

    # attach to logger so trimesh messages will be printed to console
    trimesh.util.attach_to_log()

    """
    LOAD THE MODEL:
    by default, trimesh will do a light processing, which will
    remove any NaN values and merge vertices that share position
    if you want to not do this on load, you can pass `process=False`
    """
    raw_mesh = trimesh.load(filepath)

    """
    OLD ATTEMPT
    Preview mesh in an OpenGL window (need pyglet and scipy to be installed):
    """
    # raw_mesh.show(smooth=False, line_settings={'point_size': 20, 'line_width': 1})

    """
    NEW ATTEMPT:
    Use trimesh only to load the file and pyrender to view it

    To run pyrender on MAC:
    edit PyOpenGL file in venv/lib/python3.7/site-packages/OpenGL/platform/ctypesloader.py changing line

        fullName = util.find_library( name )
    to

        fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'

    """

    mesh = pyrender.Mesh.from_trimesh(raw_mesh)

    return mesh


def view_scene(mesh):
    """
    Function used to view a mesh.
    It constructs an object scene and open the passed mesh.
    :param mesh: the mesh object that has to be displayed
    """

    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)










