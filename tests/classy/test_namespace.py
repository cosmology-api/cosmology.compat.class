"""Test the Cosmology API compat library."""

import cosmology.compat.classy as namespace
from cosmology.api import CosmologyNamespace


def test_namespace_is_compliant():
    """Test :mod:`cosmology.compat.classy.constants`."""
    assert isinstance(namespace, CosmologyNamespace)
