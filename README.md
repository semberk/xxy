# Anisotropic_Gradient_Damage #
A FEniCS Project-based code for simulating brittle fracture with anisotropic surface energy.

# Description #
Anisotropic_gradient_gamage is an open-source code that provides finite element-based
implementation of anisotropic gradient damage models, that are used for phase-field
simulation of brittle fracture phenomena with weakly and strongly anisotropic surface energy.
It is based on [Fenics-Shells](https://bitbucket.org/unilucompmech/fenics-shells/src/master/) and [FEniCS Project](http://fenicsproject.org).

# Citing #
Please cite the following work if you find this code useful for your work:

    @article{li2018variational,
	    Title = {A variational phase-field model of brittle fracture with anisotropic surface energy},
        Author = {Li, Bin and Maurini, Corrado.},
		Journal = {In Preparation},
        Volume = {},
        Number = {},
        Pages = {},
        Year = {2018}}

	@article{li2015phase,
  		Title={Phase-field modeling and simulation of fracture in brittle materials with strongly anisotropic surface energy},
  		Author={Li, Bin and Peco, Christian and Mill{\'a}n, Daniel and Arias, Irene and Arroyo, Marino},
  		Journal={International Journal for Numerical Methods in Engineering},
  		Volume={102},
  		Number={3-4},
  		Pages={711--727},
  		Year={2015},
  		Publisher={Wiley Online Library}}
        
    @article{hale2018simple,
        Title={Simple and extensible plate and shell finite element models through automatic code generation tools},
        Author={Hale, Jack and Brunetti, Matteo and Bordas, St{\'e}phane and Maurini, Corrado},
        Journal={International Journal for Numerical Methods in Engineering},
  		Volume={},
  		Number={},
  		Pages={},
        Year={2018}}
along with the appropriate general [FEniCS citations](http://fenicsproject.org/citing).

# Getting started #
*  Install FEniCS by following the instructions at

# Contributing #
We are always looking for contributions and help. If you have ideas, nice applications
or code contributions then we would be happy to help you get them included. We ask you
to follow the [FEniCS Project git workflow](https://bitbucket.org/fenics-project/dolfin/wiki/Git%20cookbook%20for%20FEniCS%20developers).


# Issues and Support #
Please use the [bugtracker](http://bitbucket.org/bin-mech/anisotropic_gradient_damage)
to report any issues.
For support or questions please email 
| `Bin Li` at <bin.li@upmc.fr>  or 
| `Corrado Maurini` at <corrado.maurini@upmc.fr>.

# Authors #
| Bin LI, Sorbonne Universite, Paris, France.
| [Corrado MAURINI](http://www.lmm.jussieu.fr/~corrado/), Sorbonne Universite, Paris, France.


# License #
anisotropic-gradient-damage is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License along
with gradient-damage.  If not, see http://www.gnu.org/licenses/.

# Notes #
HDF5File stores the output in a special way allowing for loading of data into dolfin,
but incompatible with any viewer.

* To store data for viewing, use **.xdmf** format

* To store data for reloading, use **.HDF5** format

* To store data for viewing and reloading, use **.xdmf** and **.HDF5** format

We use **gmsh** for unstructured mesh generation. Make sure that **gmsh** is in your system **PATH**.
For **multi-material** , you could assign indicators for **subdomains** and **boundaries** directly in 
the ``.geo`` file, for instance :
``Physical Line (%) = {%};``
``Physical Surface (%) = {%};``.