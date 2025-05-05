"""Test the Cosmology API compat library."""

from types import SimpleNamespace

import classy
import pytest

from cosmology import api
from cosmology.compat.classy._core import CosmologyWrapper

################################################################################
# TESTS
################################################################################


class Test_CosmologyWrapper:
    @pytest.fixture(scope="class")
    def cosmo(self) -> classy.Class:
        cosmo = classy.Class()
        cosmo.compute()
        return cosmo

    @pytest.fixture(scope="class")
    def vcosmo(self, cosmo):
        return SimpleNamespace()

    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return CosmologyWrapper(cosmo)

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that CosmologyWrapper is a CosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, api.Cosmology)
        assert isinstance(wrapper, api.CosmologyWrapper)

    def test_getattr(self, wrapper, cosmo):
        """Test that the wrapper can access the attributes of the wrapped object."""
        assert wrapper.z_reio() == cosmo.z_reio()
