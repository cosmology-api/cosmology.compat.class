"""CAMB cosmology constants.

From the :mod:`cosmology.api`, the list of required constants is:

- c: Speed of light in km s-1.
- G: Gravitational constant G in pc km2 s-2 Msol-1.
"""

import numpy as np

from cosmology.compat.classy._core import Array

__all__ = ["c", "G"]

c: Array = np.array(299792.458)  # [km s-1]
# G: CODATA 2018 value
G: Array = np.array(4.30091727003628e-3)  # [pc km2 s-2 Msol-1]
