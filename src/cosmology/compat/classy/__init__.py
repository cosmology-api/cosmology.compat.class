"""The Cosmology API compatability library for :mod:`classy`.

This library provides wrappers for CAMB cosmology objects to be compatible with
the Cosmology API. The available wrappers are:

- :class:`.StandardCosmology`: the Cosmology API wrapper for
  :mod:`classy`.


There are the following required objects for a Cosmology-API compatible library:

- constants: a module of constants. See :mod:`cosmology.compat.classy.constants`
  for details.
"""

from cosmology.compat.classy import constants
from cosmology.compat.classy._standard import StandardCosmologyWrapper

__all__ = [
    # Cosmology API
    "constants",
    # Wrappers
    "StandardCosmologyWrapper",
]
