# MR-code
Code for the project of the course: Multimedia Retrieval

Developed by:
- Di Grandi Daniele
- Wolters Matthijs

-----------

SET UP:

Due to the open3d library, the project should run with Python 3.6, 3.7 or 3.8. Latest / older versions would result in an installation error.

1) Download the zip file of the project
2) Open the whole folder (MR-code) in an editor (we coded in PyCharm, so the best option would be to open it in PyCharm)
3) Extract the zip file that contains the databases: benchmark.zip.
4) Insert the extracted benchmark folder containing the 3 databases (original: db, refined: db_refined, refined and normalised: db_ref_normalised) inside the project folder (MR-code). The 3 databases have to be inside the benchmark folder, which has to be inside the MR-code folder.
5) Create a new empty interpreter for the project (in PyCharm: go to preferences -> project interpreter -> create a new one)
6) Run the following command in the command line (be sure to be in the same folder of the project): 
   - Unix/macOS: python3 -m pip install -r requirements.txt
   - Windows: py -m pip install -r requirements.txt
   
   If for some reasons that doesn't work, manually install one by one (with pip inslall library_name or through the interpreter settings) all these libraries:
   - annoy
   - colorcet
   - matplotlib
   - mplcursors
   - numpy
   - open3d
   - pandas
   - pyglet
   - scikit-learn
   - scipy
   - seaborn
   - setuptools
   - trimesh

5) Run main.py
