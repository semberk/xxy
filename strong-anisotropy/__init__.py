# Copyright (C) 2018 Bin Li and Corrado Maurini
#
# This file is part of fenics-shells.
#
# This is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# If not, see <http://www.gnu.org/licenses/>.

"""This package implements strongly anisotropic phase-field models of fracture."""

__version__ = "1.0.0"

import dolfin as df
# We use DG everywhere, turn this on!
df.parameters["ghost_mode"] = "shared_facet"

# From fenics-shells: https://fenics-shells.readthedocs.io/en/latest/
from .fem import *
