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

# path of the file:
file = "./benchmark/db/0//m99/m99.off"

# load the model.
# by default, trimesh will do a light processing, which will
# remove any NaN values and merge vertices that share position
# if you want to not do this on load, you can pass `process=False`
mesh = trimesh.load(file)

# preview mesh in an opengl window (need pyglet and scipy to be installed)
mesh.show(smooth=False, line_settings={'point_size': 20, 'line_width': 1})

"""
fuze_trimesh = trimesh.load(file)
mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

"""



