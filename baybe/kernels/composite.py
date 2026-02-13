"""Composite kernels (that is, kernels composed of other kernels)."""

import gc
from functools import reduce
from operator import add, mul
from typing import TYPE_CHECKING

from attrs import define, field
from attrs.converters import optional as optional_c
from attrs.validators import deep_iterable, gt, instance_of, min_len
from attrs.validators import optional as optional_v
from typing_extensions import override

from baybe.kernels.base import CompositeKernel, Kernel
from baybe.priors.base import Prior
from baybe.utils.basic import to_tuple
from baybe.utils.validation import finite_float

if TYPE_CHECKING:
    import torch


@define(frozen=True)
class ScaleKernel(CompositeKernel):
    """A kernel for decorating existing kernels with an outputscale."""

    base_kernel: Kernel = field(validator=instance_of(Kernel))
    """The base kernel that is being decorated."""

    outputscale_prior: Prior | None = field(
        default=None, validator=optional_v(instance_of(Prior))
    )
    """An optional prior on the output scale."""

    outputscale_initial_value: float | None = field(
        default=None,
        converter=optional_c(float),
        validator=optional_v([finite_float, gt(0.0)]),
    )
    """An optional initial value for the output scale."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        import torch

        from baybe.utils.torch import DTypeFloatTorch

        gpytorch_kernel = super().to_gpytorch(*args, **kwargs)
        if (initial_value := self.outputscale_initial_value) is not None:
            gpytorch_kernel.outputscale = torch.tensor(
                initial_value, dtype=DTypeFloatTorch
            )
        return gpytorch_kernel


@define(frozen=True)
class AdditiveKernel(CompositeKernel):
    """A kernel representing the sum of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be summed."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(add, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


@define(frozen=True)
class ProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be multiplied."""

    @override
    def to_gpytorch(self, *args, **kwargs):
        return reduce(mul, (k.to_gpytorch(*args, **kwargs) for k in self.base_kernels))


@define(frozen=True)
class SeperableProductKernel(CompositeKernel):
    """A kernel representing the product of a collection of base kernels."""

    base_kernels: tuple[Kernel, ...] = field(
        converter=to_tuple,
        validator=deep_iterable(
            member_validator=instance_of(Kernel), iterable_validator=min_len(2)
        ),
    )
    """The individual kernels to be multiplied."""

    @override
    def to_gpytorch(
        self,
        *args,
        ard_num_dims_tup: tuple[int | None, ...] | None = None,
        active_dims_tup: tuple[tuple[int, ...] | None, ...] | None = None,
        batch_shape: torch.Size | None = None,
        **kwargs,
    ):
        assert ard_num_dims_tup is None or len(ard_num_dims_tup) == len(
            self.base_kernels
        ), "Ard num dims must be specified for every base kernel."

        assert active_dims_tup is None or (
            len(active_dims_tup) == len(self.base_kernels)
            and all(
                item is None
                or (isinstance(item, tuple) and all(isinstance(x, int) for x in item))
                for item in active_dims_tup
            )
        ), "Active dims must be specified for every base kernel."

        return reduce(
            mul,
            (
                k.to_gpytorch(
                    *args,
                    ard_num_dims=ard_num_dims_tup[k],
                    active_dims=active_dims_tup[k],
                    batch_shape=batch_shape,
                    **kwargs,
                )
                for k in self.base_kernels
            ),
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
