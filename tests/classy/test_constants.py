"""Test the Cosmology API compat library."""

from cosmology import api
from cosmology.compat.classy import constants


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.classy.constants`."""
    assert isinstance(constants, api.CosmologyConstantsNamespace)
