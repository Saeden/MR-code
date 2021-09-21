"""
Install:

trimesh
pyglet
scipy
pyrender

"""

import trimesh
import pyrender


# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

"""
FILE PATH:
"""
file = "./benchmark/db/0//m99/m99.off"

"""
LOAD THE MODEL:
by default, trimesh will do a light processing, which will
remove any NaN values and merge vertices that share position
if you want to not do this on load, you can pass `process=False`
"""
#mesh = trimesh.load(file)

"""
Preview mesh in an OpenGL window (need pyglet and scipy to be installed):
"""
#mesh.show(smooth=False, line_settings={'point_size': 20, 'line_width': 1})


"""
NEW ATTEMPT:
Use trimesh only to load the file and pyrender to view it

To run pyrender on MAC:
edit PyOpenGL file in venv/lib/python3.7/site-packages/OpenGL/platform/ctypesloader.py changing line

    fullName = util.find_library( name )
to

    fullName = '/System/Library/Frameworks/OpenGL.framework/OpenGL'
    
"""
fuze_trimesh = trimesh.load(file)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)




