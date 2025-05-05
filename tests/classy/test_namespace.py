"""Test the Cosmology API compat library."""

import cosmology.compat.classy as namespace
from cosmology import api


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.classy.constants`."""
    assert isinstance(namespace, api.CosmologyNamespace)
