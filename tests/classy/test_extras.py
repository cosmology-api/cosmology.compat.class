"""Test the Cosmology API compat library."""

from __future__ import annotations

import numpy as np
from hypothesis import given

from cosmology.api import CriticalDensity, HubbleParameter
from cosmology.compat.classy import constants

from .conftest import z_arr_st

################################################################################
# TESTS
################################################################################


class CriticalDensity_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, CriticalDensity)

    def test_critical_density0(self, wrapper, cosmo):
        """
        Test that the wrapper's critical_density0 is the same as
        critical_density0.
        """
        expect = (
            3e6 * constants.c**2 * cosmo.Hubble(0) ** 2 / (8 * np.pi * constants.G)
        )
        assert np.allclose(wrapper.critical_density0, expect)
        assert isinstance(wrapper.critical_density0, np.ndarray)

    @given(z_arr_st(max_value=1e9))
    def test_critical_density(self, wrapper, vcosmo, z):
        r"""Test that the wrapper's critical_density is critical_density."""
        rho = wrapper.critical_density(z)
        expect = (
            3e6 * constants.c**2 * vcosmo.Hubble(z) ** 2 / (8 * np.pi * constants.G)
        )
        assert np.allclose(rho, expect)
        assert isinstance(rho, np.ndarray)


class HubbleParameter_Test:
    def test_wrapper_is_compliant(self, wrapper):
        """Test that AstropyCosmology is a BackgroundCosmologyWrapper."""
        if hasattr(super(), "test_wrapper_is_compliant"):
            super().test_wrapper_is_compliant(wrapper)

        assert isinstance(wrapper, HubbleParameter)

    def test_H0(self, wrapper, cosmo):
        """Test that the wrapper has the same H0 as the wrapped object."""
        assert wrapper.H0 == constants.c * cosmo.Hubble(0)  # noqa: SIM300
        assert isinstance(wrapper.H0, np.ndarray)

    def test_hubble_distance(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_distance as the wrapped object."""
        assert wrapper.hubble_distance == 1 / cosmo.Hubble(0)
        assert isinstance(wrapper.hubble_distance, np.ndarray)

    def test_hubble_time(self, wrapper, cosmo):
        """Test that the wrapper has the same hubble_time as the wrapped object."""
        expect = np.float64("978.5") / cosmo.Hubble(0) / constants.c
        assert np.allclose(wrapper.hubble_time, expect)
        assert isinstance(wrapper.hubble_time, np.ndarray)

    @given(z_arr_st(min_value=0, max_value=1e10))
    def test_H(self, wrapper, vcosmo, z):
        """Test that the wrapper's H is the same as the wrapped object's."""
        H = wrapper.H(z)
        assert np.array_equal(H, constants.c * vcosmo.Hubble(z))
        assert isinstance(H, np.ndarray)

    @given(z_arr_st(min_value=0, max_value=1e10))
    def test_H_over_H0(self, wrapper, vcosmo, z):
        """Test that the wrapper's efunc is the same as the wrapped object's."""
        e = wrapper.H_over_H0(z)
        assert np.array_equal(e, vcosmo.Hubble(z) / vcosmo.Hubble(0))
        assert isinstance(e, np.ndarray)
