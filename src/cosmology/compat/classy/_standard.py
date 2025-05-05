"""The Cosmology API compatibility wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, overload

import numpy as np
from numpy import vectorize
from scipy.interpolate import InterpolatedUnivariateSpline

from cosmology.compat.classy import constants
from cosmology.compat.classy._core import Array, CosmologyWrapper, InputT

__all__: list[str] = []


_MPCS_KM_TO_GYR = np.array("978.5", dtype=np.float64)  # [Mpc s / km -> Gyr]


@dataclass(frozen=True)
class StandardCosmologyWrapper(CosmologyWrapper):
    """FLRW Cosmology API wrapper for CAMB cosmologies."""

    def __post_init__(self) -> None:
        """Run-time post-processing.

        Note that if this module is c-compiled (e.g. with :mod:`mypyc`) that
        the type of ``self.cosmo`` must be ``CAMBdata`` at object creation
        and cannot be later processed here.
        """
        super().__post_init__()

        bkg = self.cosmo.get_background()
        z = bkg["z"][::-1]

        # Sum over all ncdm species
        rho_ncdm = np.zeros_like(bkg["(.)rho_ncdm[0]"])
        i = 0
        while f"(.)rho_ncdm[{i}]" in bkg:
            rho_ncdm += bkg[f"(.)rho_ncdm[{i}]"]
            i += 1

        # Redshift-dependent Omega_nu
        Omega_nu_z = rho_ncdm / bkg["(.)rho_crit"]
        object.__setattr__(self, "_Omega_nu0", Omega_nu_z[-1])

        m_nu = self.cosmo.cosmo_arguments().get("m_ncdm")
        m_nu_arr = () if m_nu is None else tuple(np.array(m_nu.split(","), dtype=float))
        object.__setattr__(self, "_m_nu", m_nu_arr)


        self._cosmo_fn: dict[str, Any]
        object.__setattr__(
            self,
            "_cosmo_fn",
            {
                "Om_m": vectorize(self.cosmo.Om_m),
                "Hubble": vectorize(self.cosmo.Hubble),
                "angular_distance": vectorize(self.cosmo.angular_distance),
                "luminosity_distance": vectorize(self.cosmo.luminosity_distance),
                "comoving_distance": InterpolatedUnivariateSpline(
                    z,
                    bkg["comov. dist."][::-1],
                    k=3,
                    ext=2,
                    check_finite=True,
                ),
                "inv_comoving_distance": InterpolatedUnivariateSpline(
                    bkg["comov. dist."][::-1],
                    z,
                    k=3,
                    ext=2,
                    check_finite=True,
                ),
                "Omega_nu_interp": InterpolatedUnivariateSpline(z, Omega_nu_z[::-1], k=3)
            },
        )

    # ----------------------------------------------
    # TotalComponent

    @property
    def Omega_tot0(self) -> Array:
        r"""Omega total; the total density/critical density at z=0.

        Note this should alway be 1.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm r} + \Omega_{\rm de}
            + \Omega_{\rm k}
        """
        return np.array(
            self.cosmo.Omega_Lambda()
            + self.cosmo.Omega_m()
            + self.cosmo.Omega0_k()
            + self.cosmo.Omega_r()
        )

    def Omega_tot(self, z: InputT, /) -> Array:
        r"""Redshift-dependent total density parameter.

        This is the sum of the matter, radiation, neutrino, dark energy, and
        curvature density parameters.

        .. math::

            \Omega_{\rm tot} = \Omega_{\rm m} + \Omega_{\rm \gamma} +
            \Omega_{\rm \nu} + \Omega_{\rm de} + \Omega_{\rm k}
        """
        raise NotImplementedError
        #  Basically just return np.ones_like(z)

    # ----------------------------------------------
    # CurvatureComponent

    @property
    def Omega_k0(self) -> Array:
        """Omega curvature; the effective curvature density/critical density at z=0."""
        return np.asarray(self.cosmo.Omega0_k())

    def Omega_k(self, z: InputT, /) -> Array:
        """Redshift-dependent curvature density parameter."""
        raise NotImplementedError

    # ----------------------------------------------
    # MatterComponent

    @property
    def Omega_m0(self) -> Array:
        """Matter density at z=0."""
        return np.asarray(self.cosmo.Omega_m())

    def Omega_m(self, z: InputT, /) -> Array:
        """Redshift-dependent non-relativistic matter density parameter.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest; see `Onu`.

        """
        return np.asarray(self._cosmo_fn["Om_m"](z))

    # ----------------------------------------------
    # BaryonComponent

    @property
    def Omega_b0(self) -> Array:
        """Baryon density at z=0."""
        return np.asarray(self.cosmo.Omega_b())

    def Omega_b(self, z: InputT, /) -> Array:
        """Redshift-dependent baryon density parameter.

        Raises
        ------
        ValueError
            If ``Ob0`` is `None`.

        """
        raise NotImplementedError

    # ----------------------------------------------
    # NeutrinoComponent

    @property
    def Omega_nu0(self) -> Array:
        """Omega nu; the density/critical density of neutrinos at z=0."""
        return np.asarray(self._Omega_nu0)

    @property
    def Neff(self) -> Array:
        """Effective number of neutrino species."""
        return np.asarray(self.cosmo.Neff())

    @property
    def m_nu(self) -> tuple[Array, ...]:
        """Neutrino mass in eV."""
        return self._m_nu

    def Omega_nu(self, z: InputT, /) -> Array:
        r"""Redshift-dependent neutrino density parameter."""
        return np.asarray(self._Omega_nu_interp(z))

    # ----------------------------------------------
    # DarkEnergyComponent

    @property
    def Omega_de0(self) -> Array:
        """Dark energy density at z=0."""
        return np.asarray(self.cosmo.Omega_Lambda())

    def Omega_de(self, z: InputT, /) -> Array:
        """Redshift-dependent dark energy density parameter."""
        raise NotImplementedError

    # ----------------------------------------------
    # DarkMatterComponent

    @property
    def Omega_dm0(self) -> Array:
        """Omega dark matter; dark matter density/critical density at z=0."""
        return np.asarray(self.cosmo.Omega0_cdm())

    def Omega_dm(self, z: InputT, /) -> Array:
        """Redshift-dependent dark matter density parameter.

        Notes
        -----
        This does not include neutrinos, even if non-relativistic at the
        redshift of interest.

        """
        raise NotImplementedError

    # ----------------------------------------------
    # PhotonComponent

    @property
    def Omega_gamma0(self) -> Array:
        """Omega gamma; the density/critical density of photons at z=0."""
        return np.asarray(self.cosmo.Omega_g())

    def Omega_gamma(self, z: InputT, /) -> Array:
        """Redshift-dependent photon density parameter."""
        return self.Omega_gamma0 * (z + 1.0) ** 4 / self.H_over_H0(z) ** 2

    # ----------------------------------------------
    # CriticalDensity

    @property
    def critical_density0(self) -> Array:
        """Critical density at z = 0 in Msol Mpc-3."""
        return np.array(3e6 * self.H0**2 / (8 * np.pi * constants.G))

    def critical_density(self, z: InputT, /) -> Array:
        """Redshift-dependent critical density in Msol Mpc-3."""
        return np.array(3e6 * self.H(z) ** 2 / (8 * np.pi * constants.G))

    # ----------------------------------------------
    # HubbleParameter

    @property
    def H0(self) -> Array:
        """Hubble constant at z=0 in km s-1 Mpc-1."""
        return np.array(constants.c * self.cosmo.Hubble(0))

    @property
    def hubble_distance(self) -> Array:
        """Hubble distance in Mpc."""
        return np.array(1 / self.cosmo.Hubble(0))

    @property
    def hubble_time(self) -> Array:
        """Hubble time in Gyr."""
        return np.array(_MPCS_KM_TO_GYR / self.H0)

    def H(self, z: InputT, /) -> Array:
        """Hubble function :math:`H(z)` in km s-1 Mpc-1."""
        return np.array(constants.c * self._cosmo_fn["Hubble"](z))

    def H_over_H0(self, z: InputT, /) -> Array:
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
        return self._cosmo_fn["Hubble"](z) / self.cosmo.Hubble(0)

    # ----------------------------------------------
    # Scale factor

    @property
    def scale_factor0(self) -> Array:
        """Scale factor at z=0."""
        return np.asarray(1.0)

    def scale_factor(self, z: InputT, /) -> Array:
        """Redshift-dependenct scale factor :math:`a = a_0 / (1 + z)`."""
        return np.asarray(self.scale_factor0 / (z + 1))

    # ----------------------------------------------
    # Temperature

    @property
    def T_cmb0(self) -> Array:
        """Temperature of the CMB at z=0."""
        return np.asarray(self.cosmo.T_cmb())

    def T_cmb(self, z: InputT, /) -> Array:
        """Temperature of the CMB at redshift ``z``."""
        return self.T_cmb0 * (z + 1)

    # ----------------------------------------------
    # Time

    def age(self, z: InputT, /) -> Array:
        """Age of the universe in Gyr at redshift ``z``."""
        raise NotImplementedError

    # ----------------------------------------------
    # Comoving distance

    @overload
    def comoving_distance(self, z: InputT, /) -> Array: ...

    @overload
    def comoving_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def comoving_distance(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        r"""Comoving line-of-sight distance :math:`d_c(z)` in Mpc.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_c(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_c(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The comoving distance :math:`d_c` in Mpc.

        """
        z1, z2 = (0, z1) if z2 is None else (z1, z2)
        return self._cosmo_fn["comoving_distance"](z2) - self._cosmo_fn[
            "comoving_distance"
        ](z1)

    def inv_comoving_distance(self, dc: InputT, /) -> Array:
        """Inverse comoving distance :math:`d_c^{-1}(d)` in Mpc^-1.

        This is the inverse of the comoving distance function. It is used to
        calculate the comoving distance from a given distance.

        Parameters
        ----------
        dc : Array, positional-only
            Input distances.

        Returns
        -------
        Array
            The inverse comoving distance :math:`d_c^{-1}` in Mpc^-1.

        """
        return self._cosmo_fn["inv_comoving_distance"](dc)

    @overload
    def transverse_comoving_distance(self, z: InputT, /) -> Array: ...

    @overload
    def transverse_comoving_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def transverse_comoving_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> Array:
        r"""Transverse comoving distance :math:`d_M(z)` in Mpc.

        This value is the transverse comoving distance at redshift ``z``
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_M(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_M(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The comoving transverse distance :math:`d_M` in Mpc.

        """
        raise NotImplementedError

    def _comoving_volume_flat(self, z: InputT, /) -> Array:
        return 4.0 / 3.0 * np.pi * self.comoving_distance(z) ** 3

    def _comoving_volume_positive(self, z: InputT, /) -> Array:
        dh = self.hubble_distance
        x = self.transverse_comoving_distance(z) / dh
        term1 = 4.0 * np.pi * dh**3 / (2.0 * self.Omega_k0)
        term2 = x * np.sqrt(1 + self.Omega_k0 * (x) ** 2)
        term3 = np.sqrt(np.abs(self.Omega_k0)) * x

        return term1 * (
            term2 - 1.0 / np.sqrt(np.abs(self.Omega_k0)) * np.arcsinh(term3)
        )

    def _comoving_volume_negative(self, z: InputT, /) -> Array:
        dh = self.hubble_distance
        x = self.transverse_comoving_distance(z) / dh
        term1 = 4.0 * np.pi * dh**3 / (2.0 * self.Omega_k0)
        term2 = x * np.sqrt(1 + self.Omega_k0 * (x) ** 2)
        term3 = np.sqrt(np.abs(self.Omega_k0)) * x
        return term1 * (term2 - 1.0 / np.sqrt(np.abs(self.Omega_k0)) * np.arcsin(term3))

    @overload
    def comoving_volume(self, z: InputT, /) -> Array: ...

    @overload
    def comoving_volume(self, z1: InputT, z2: InputT, /) -> Array: ...

    def comoving_volume(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        r"""Comoving volume in cubic Mpc.

        This is the volume of the universe encompassed by redshifts less than
        ``z``. For the case of :math:`\Omega_k = 0` it is a sphere of radius
        `comoving_distance` but it is less intuitive if :math:`\Omega_k` is not.
        """
        if z2 is not None:
            raise NotImplementedError

        if self.Omega_k0 == 0:
            cv = self._comoving_volume_flat(z1)
        elif self.Omega_k0 > 0:
            cv = self._comoving_volume_positive(z1)
        else:
            cv = self._comoving_volume_negative(z1)
        return cv

    def differential_comoving_volume(self, z: InputT, /) -> Array:
        r"""Differential comoving volume in cubic Mpc per steradian.

        If :math:`V_c` is the comoving volume of a redshift slice with solid
        angle :math:`\Omega`, this function ...

        .. math::

            \mathtt{dvc(z)}
            = \frac{1}{d_H^3} \, \frac{dV_c}{d\Omega \, dz}
            = \frac{x_M^2(z)}{E(z)}
            = \frac{\mathtt{xm(z)^2}}{\mathtt{ef(z)}} \;.

        """
        return (
            self.transverse_comoving_distance(z) / self.hubble_distance
        ) ** 2 / self.H_over_H0(z)

    # ----------------------------------------------
    # Proper

    @overload
    def proper_distance(self, z: InputT, /) -> Array: ...

    @overload
    def proper_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def proper_distance(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        r"""Proper distance :math:`d` in Mpc.

        The proper distance is the distance between two objects at redshifts
        ``z1`` and ``z2``, including the effects of the expansion of the
        universe.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The proper distance :math:`d` in Mpc.

        """
        raise NotImplementedError

    @overload
    def proper_time(self, z: InputT, /) -> Array: ...

    @overload
    def proper_time(self, z1: InputT, z2: InputT, /) -> Array: ...

    def proper_time(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        r"""Proper time :math:`t` in Gyr.

        The proper time is the proper distance divided by
        :attr:`~cosmology.api.CosmologyConstantsNamespace.c`.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`t(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`t(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The proper time :math:`t` in Gyr.

        """
        raise NotImplementedError

    # ----------------------------------------------

    @overload
    def lookback_distance(self, z: InputT, /) -> Array: ...

    @overload
    def lookback_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def lookback_distance(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        r"""Lookback distance :math:`d_T` in Mpc.

        The lookback distance is the subjective distance it took light to travel
        from redshift ``z1`` to  ``z2``.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_T(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_T(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The lookback distance :math:`d_T` in Mpc.

        """
        raise NotImplementedError

    @overload
    def lookback_time(self, z: InputT, /) -> Array: ...

    @overload
    def lookback_time(self, z1: InputT, z2: InputT, /) -> Array: ...

    def lookback_time(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        """Lookback time in Gyr.

        The lookback time is the time that it took light from being emitted at
        one redshift to being observed at another redshift. Effectively it is the
        difference between the age of the Universe at the two redshifts.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the time
            :math:`t_T(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the time :math:`t_T(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The lookback time in Gyr.

        """
        raise NotImplementedError

    # ----------------------------------------------
    # Angular diameter

    @overload
    def angular_diameter_distance(self, z: InputT, /) -> Array: ...

    @overload
    def angular_diameter_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def angular_diameter_distance(
        self, z1: InputT, z2: InputT | None = None, /
    ) -> Array:
        """Angular diameter distance :math:`d_A` in Mpc.

        This gives the proper (sometimes called 'physical') transverse distance
        corresponding to an angle of 1 radian for an object at redshift ``z``
        ([1]_, [2]_, [3]_).

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_A(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_A(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The angular diameter distance :math:`d_A` in Mpc.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 421-424.
        .. [2] Weedman, D. (1986). Quasar astronomy, pp 65-67.
        .. [3] Peebles, P. (1993). Principles of Physical Cosmology, pp 325-327.

        """
        if z2 is not None:
            raise NotImplementedError
        return np.asarray(self._cosmo_fn["angular_distance"](z1))

    # ----------------------------------------------
    # Luminosity distance

    @overload
    def luminosity_distance(self, z: InputT, /) -> Array: ...

    @overload
    def luminosity_distance(self, z1: InputT, z2: InputT, /) -> Array: ...

    def luminosity_distance(self, z1: InputT, z2: InputT | None = None, /) -> Array:
        """Redshift-dependent luminosity distance :math:`d_L` in Mpc.

        This is the distance to use when converting between the bolometric flux
        from an object at redshift ``z`` and its bolometric luminosity [1]_.

        Parameters
        ----------
        z : Array, positional-only
        z1, z2 : Array, positional-only
            Input redshifts. If one argument ``z`` is given, the distance
            :math:`d_L(0, z)` is returned. If two arguments ``z1, z2`` are
            given, the distance :math:`d_L(z_1, z_2)` is returned.

        Returns
        -------
        Array
            The luminosity distance :math:`d_L` in Mpc.

        References
        ----------
        .. [1] Weinberg, 1972, pp 420-424; Weedman, 1986, pp 60-62.

        """
        if z2 is not None:
            raise NotImplementedError
        return np.asarray(self._cosmo_fn["luminosity_distance"](z1))
