"""Test the Cosmology API compat library."""


def test_imported():
    """This is a namespace package, so it should be importable."""
    import cosmology.compat.classy

    assert cosmology.compat.classy.__name__ == "cosmology.compat.classy"
