"""Compact influence functions for intersection-driven displacement data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class WeightingFunction:
    """Protocol-like base class for normalized distance weights.

    A weighting function maps a normalized distance to an influence weight.
    "Normalized" means the caller has already divided the physical distance by
    the influence radius, so the unit interval spans full influence at ``0.0``
    to the edge of support at ``1.0``. Subclasses accept array input and return
    an array of the same shape.

    Subclasses in this module all have compact support: the weight is exactly
    zero at and beyond a normalized distance of one, so a source cannot
    influence a point outside its radius. Calling this base class directly
    raises :exc:`NotImplementedError`.
    """

    def __call__(self, normalized_distance):
        """Raise, since the base class defines no weight.

        Parameters
        ----------
        normalized_distance
            Distance already divided by the influence radius.

        Raises
        ------
        NotImplementedError
            Always; subclasses provide the weight.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class InverseDistanceWeighting(WeightingFunction):
    """Compact inverse-distance weight.

    The usual singular IDW expression is multiplied by a compact smoothstep
    envelope and normalized to one at the source.  Consequently the returned
    value is one at zero distance and exactly zero at and beyond unit distance.

    Attributes
    ----------
    exponent
        Power applied to the ``1 / (1 + d)`` inverse term. Larger values
        concentrate influence nearer the source.
    epsilon
        Distance at or below which the weight is forced to exactly ``1.0``,
        avoiding the singularity at the source.
    """

    exponent: float = 2.0
    epsilon: float = 1e-12

    def __call__(self, normalized_distance):
        """Return the compact inverse-distance weight.

        Parameters
        ----------
        normalized_distance
            Distance already divided by the influence radius, as a scalar or
            array.

        Returns
        -------
        numpy.ndarray
            Weights with the shape of the input: exactly ``1.0`` at or below
            ``epsilon``, the enveloped inverse-distance value inside the unit
            interval, and exactly ``0.0`` at and beyond unit distance.
        """
        distance = np.asarray(normalized_distance, dtype=float)
        clipped = np.clip(distance, 0.0, 1.0)
        envelope = 1.0 - clipped * clipped * (3.0 - 2.0 * clipped)
        inverse = (1.0 / (1.0 + clipped)) ** float(self.exponent)
        result = envelope * inverse
        return np.where(
            distance <= self.epsilon,
            1.0,
            np.where(distance < 1.0, result, 0.0),
        )


@dataclass(frozen=True)
class GaussianWeighting(WeightingFunction):
    """Truncated Gaussian weight on a unit-radius support.

    Evaluates ``exp(-sharpness * d**2)`` inside the support and returns exactly
    zero at and beyond unit distance. The truncation leaves a genuine jump of
    ``exp(-sharpness)`` at ``d == 1``, which is nonzero for **every** finite
    ``sharpness``: raising ``sharpness`` shrinks the jump until it is
    numerically negligible but never removes it. Unlike the other weights in
    this module, this one is therefore not continuous at the support boundary.

    Attributes
    ----------
    sharpness
        Coefficient of the squared normalized distance in the exponent. Larger
        values decay faster and shrink, without eliminating, the jump at the
        truncation radius.
    """

    sharpness: float = 4.0

    def __call__(self, normalized_distance):
        """Return the truncated Gaussian weight.

        Parameters
        ----------
        normalized_distance
            Distance already divided by the influence radius, as a scalar or
            array.

        Returns
        -------
        numpy.ndarray
            Weights with the shape of the input: ``exp(-sharpness * d**2)``
            inside the unit interval and exactly ``0.0`` at and beyond unit
            distance.
        """
        distance = np.asarray(normalized_distance, dtype=float)
        value = np.exp(-float(self.sharpness) * distance**2)
        return np.where(distance < 1.0, value, 0.0)


@dataclass(frozen=True)
class LinearWeighting(WeightingFunction):
    """Linear decay on a unit-radius support.

    Returns ``1 - d`` clipped to the unit interval, so the weight falls
    linearly from one at the source to zero at unit distance and stays zero
    beyond it. It carries no parameters.
    """

    def __call__(self, normalized_distance):
        """Return the linearly decaying weight.

        Parameters
        ----------
        normalized_distance
            Distance already divided by the influence radius, as a scalar or
            array.

        Returns
        -------
        numpy.ndarray
            Weights with the shape of the input, falling linearly from ``1.0``
            at the source to ``0.0`` at unit distance and clipped to that range
            outside it.
        """
        return np.clip(1.0 - np.asarray(normalized_distance, dtype=float), 0.0, 1.0)


__all__ = [
    "GaussianWeighting",
    "InverseDistanceWeighting",
    "LinearWeighting",
    "WeightingFunction",
]
