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


Array: TypeAlias = NDArray[floating[Any]]
InputT: TypeAlias = Union[Array, float]


@dataclass(frozen=True)
class CosmologyWrapper(CosmologyWrapperAPI[Array, InputT]):
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

    @property
    def __cosmology_namespace__(self) -> CosmologyNamespace:
        """Returns :class:`~cosmology.api.CosmologyNamespace` for CLASS."""
        import cosmology.compat.classy as namespace

        return cast(CosmologyNamespace, namespace)
