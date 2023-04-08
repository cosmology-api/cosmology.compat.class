"""The Cosmology API compatability wrapper for CAMB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Union, cast

import classy
from numpy import floating
from numpy.typing import NDArray

from cosmology.api import CosmologyNamespace
from cosmology.api.compat import CosmologyWrapper as CosmologyWrapperAPI

__all__: list[str] = []

if TYPE_CHECKING:
    from typing_extensions import TypeAlias


NDFloating: TypeAlias = NDArray[floating[Any]]
InputT: TypeAlias = Union[NDFloating, float]


@dataclass(frozen=True)
class CosmologyWrapper(CosmologyWrapperAPI[NDFloating, InputT]):
    """The Cosmology API wrapper for :mod:`classy`."""

    cosmo: classy.Class
    name: str | None = None

    def __post_init__(self) -> None:
        """Run-time post-processing.

        Note that if this module is c-compiled (e.g. with :mod:`mypyc`) that
        the type of ``self.cosmo`` must be ``CAMBdata`` at object creation
        and cannot be later processed here.
        """
        if not isinstance(self.cosmo, classy.Class):
            msg = f"cosmo must be a <classy.Class>, not {type(self.cosmo)}"
            raise TypeError(msg)

        self._cosmo_fn: dict[str, Any]
        object.__setattr__(self, "_cosmo_fn", {})

    def __cosmology_namespace__(
        self, /, *, api_version: str | None = None
    ) -> CosmologyNamespace:
        """Returns an object that has all the cosmology API functions on it.

        Parameters
        ----------
        api_version: Optional[str]
            string representing the version of the cosmology API specification
            to be returned, in ``'YYYY.MM'`` form, for example, ``'2020.10'``.
            If ``None``, it return the namespace corresponding to latest version
            of the cosmology API specification.  If the given version is invalid
            or not implemented for the given module, an error is raised.
            Default: ``None``.

            .. note:: currently only `None` is supported.

        Returns
        -------
        `CosmologyNamespace`
            An object representing the CAMB cosmology API namespace.
        """
        import cosmology.compat.classy as namespace

        return cast(CosmologyNamespace, namespace)
