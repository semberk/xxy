from fenics import *
import os
import subprocess
from dolfin_utils.meshconvert import meshconvert

# Generate a XDMF/HDF5 based mesh from a Gmsh string
def mesher(geofile, meshname):

    subdir = "meshes/"
    _mesh  = Mesh() #creat empty mesh object
    if not os.path.isfile(subdir + meshname + ".xdmf"):

        if MPI.rank(mpi_comm_world()) == 0:

            # Create temporary .geo file defining the mesh
            if os.path.isdir(subdir) == False:
                os.mkdir(subdir)
            fgeo = open(subdir + meshname + ".geo", "w")
            fgeo.writelines(geofile)
            fgeo.close()

            # Calling gmsh and dolfin-convert to generate the .xml mesh (as well as a MeshFunction file)
            try:
                subprocess.call(["gmsh", "-2", "-o", subdir + meshname + ".msh", subdir + meshname + ".geo"])
            except OSError:
                print("-----------------------------------------------------------------------------")
                print(" Error: unable to generate the mesh using gmsh")
                print(" Make sure that you have gmsh installed and have added it to your system PATH")
                print("-----------------------------------------------------------------------------")
                return
            meshconvert.convert2xml(subdir + meshname + ".msh", subdir + meshname + ".xml", "gmsh")
            # Calling meshio-convert to convert GMSH format to XDMF format (efficient for large mesh)
            # try:
                # subprocess.call(["meshio-convert", subdir + meshname + ".msh", subdir + meshname + ".xdmf"])
            # except OSError:
                # print("-----------------------------------------------------------------------------")
                # print(" Error: unable to convert MSH to XDMF using meshio-convert")
                # print(" Make sure that you have meshio installed ( python3 -m pip install meshio )")
                # print("-----------------------------------------------------------------------------")

        MPI.barrier(mpi_comm_world())

        if not os.path.isfile(subdir + meshname + ".xdmf"):
            mesh = Mesh(subdir + meshname + ".xml")
            XDMF = XDMFFile(mpi_comm_world(), subdir + meshname + ".xdmf")
            XDMF.write(mesh)
            XDMF.read(_mesh)
        else:
            XDMF = XDMFFile(mpi_comm_world(), subdir + meshname + ".xdmf")
            XDMF.read(_mesh)

        if os.path.isfile(subdir + meshname + "_physical_region.xml") and os.path.isfile(subdir + meshname + "_facet_region.xml"):

            if MPI.rank(mpi_comm_world()) == 0:

                mesh = Mesh(subdir + meshname + ".xml")
                subdomains = MeshFunction("size_t", mesh, subdir + meshname + "_physical_region.xml")
                boundaries = MeshFunction("size_t", mesh, subdir + meshname + "_facet_region.xml")
                HDF5 = HDF5File(mesh.mpi_comm(), subdir + meshname + "_physical_facet.h5", "w")
                HDF5.write(mesh, "/mesh")
                HDF5.write(subdomains, "/subdomains")
                HDF5.write(boundaries, "/boundaries")

                print("Finish writting physical_facet to HDF5")

        if MPI.rank(mpi_comm_world()) == 0:

            # Keep only the .xdmf mesh
            os.remove(subdir + meshname + ".geo")
            os.remove(subdir + meshname + ".msh")
            os.remove(subdir + meshname + ".xml")

            # Info
            print("Mesh completed")

    # Read the mesh if existing
    else:
        XDMF = XDMFFile(mpi_comm_world(), subdir + meshname + ".xdmf")
        XDMF.read(_mesh)

    return _mesh