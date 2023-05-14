"""Test the Cosmology API compat library."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cosmology.api import StandardCosmology
from cosmology.api import StandardCosmologyWrapper as StandardCosmologyWrapperAPI
from cosmology.compat.classy import StandardCosmologyWrapper

from .test_components import (
    BaryonComponent_Test,
    CurvatureComponent_Test,
    DarkEnergyComponent_Test,
    DarkMatterComponent_Test,
    MatterComponent_Test,
    NeutrinoComponent_Test,
    PhotonComponent_Test,
    TotalComponent_Test,
)
from .test_core import Test_CosmologyWrapper
from .test_distances import DistanceMeasures_Test
from .test_extras import CriticalDensity_Test, HubbleParameter_Test

################################################################################
# TESTS
################################################################################


class Test_StandardCosmologyWrapper(
    TotalComponent_Test,
    CurvatureComponent_Test,
    MatterComponent_Test,
    BaryonComponent_Test,
    NeutrinoComponent_Test,
    DarkEnergyComponent_Test,
    DarkMatterComponent_Test,
    PhotonComponent_Test,
    CriticalDensity_Test,
    HubbleParameter_Test,
    DistanceMeasures_Test,
    Test_CosmologyWrapper,
):
    @pytest.fixture(scope="class")
    def wrapper(self, cosmo):
        return StandardCosmologyWrapper(cosmo)

    @pytest.fixture(scope="class")
    def vcosmo(self, cosmo):
        vc = SimpleNamespace()
        vc.Om_m = np.vectorize(cosmo.Om_m)
        vc.Hubble = np.vectorize(cosmo.Hubble)
        vc.angular_distance = np.vectorize(cosmo.angular_distance)
        vc.luminosity_distance = np.vectorize(cosmo.luminosity_distance)
        return vc

    # =========================================================================
    # Tests

    def test_wrapper_is_compliant(self, wrapper):
        """Test that StandardCosmologyWrapper is a StandardCosmologyWrapper."""
        super().test_wrapper_is_compliant(wrapper)

        # FIXME: it should be an instance
        with pytest.raises(NotImplementedError):
            assert isinstance(wrapper, StandardCosmology)

        # FIXME: it should be an instance
        with pytest.raises(NotImplementedError):
            assert isinstance(wrapper, StandardCosmologyWrapperAPI)
