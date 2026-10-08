"""Differentiable cruise fuel burn from the classical Breguet relation."""

from __future__ import annotations

from dataclasses import dataclass

import csdl_alpha as csdl
import numpy as np


@dataclass(frozen=True)
class FuelBurnParameters:
    """Aircraft and mission parameters for cruise fuel burn.

    Parameters
    ----------
    design_range_m
        Cruise design range in meters.
    thrust_specific_fuel_consumption_kg_per_newton_second
        Jet-engine thrust-specific fuel consumption in kilograms per
        newton-second.
    cruise_speed_m_s
        Cruise true airspeed in meters per second.
    initial_weight_newton
        Aircraft weight at the start of the cruise segment in newtons.
    gravity_m_s2
        Gravitational acceleration in meters per second squared. The default
        is the conventional standard gravity, exactly 9.80665 m/s².

    Notes
    -----
    Mission and aircraft values have no package defaults: callers must provide
    and source them for the vehicle and flight segment being modeled. The
    implementation is the classical Breguet jet range equation solved for
    burned fuel weight.
    """

    design_range_m: float
    thrust_specific_fuel_consumption_kg_per_newton_second: float
    cruise_speed_m_s: float
    initial_weight_newton: float
    gravity_m_s2: float = 9.80665

    def __post_init__(self) -> None:
        """Require physically positive mission constants."""
        for name in (
            "design_range_m",
            "thrust_specific_fuel_consumption_kg_per_newton_second",
            "cruise_speed_m_s",
            "initial_weight_newton",
            "gravity_m_s2",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive.")


def _variable(name: str, value: float | csdl.Variable) -> csdl.Variable:
    """Promote a scalar constant without detaching an existing CSDL variable."""
    if isinstance(value, csdl.Variable):
        return value
    return csdl.Variable(name=name, value=np.array([float(value)], dtype=float))


def compute_fuel_burn(
    lift_coefficient: float | csdl.Variable,
    drag_coefficient: float | csdl.Variable,
    parameters: FuelBurnParameters,
) -> csdl.Variable:
    """Compute cruise fuel-burn weight with analytic CSDL derivatives.

    Parameters
    ----------
    lift_coefficient
        Dimensionless aircraft lift coefficient.
    drag_coefficient
        Dimensionless total aircraft drag coefficient, including induced and
        parasite contributions.
    parameters
        Range, fuel-consumption, speed, weight, and gravity values with units
        documented by :class:`FuelBurnParameters`.

    Returns
    -------
    csdl.Variable
        Fuel weight burned over the cruise segment, in newtons.

    Notes
    -----
    For the Breguet jet relation,
    ``R = V / (g * TSFC) * (CL / CD) * log(W_initial / W_final)``.
    This function solves that expression for ``W_initial - W_final``.
    """
    lift = _variable("fuel_burn_lift_coefficient", lift_coefficient)
    drag = _variable("fuel_burn_drag_coefficient", drag_coefficient)
    exponent = (
        -parameters.design_range_m
        * parameters.gravity_m_s2
        * parameters.thrust_specific_fuel_consumption_kg_per_newton_second
        * drag
        / (parameters.cruise_speed_m_s * lift)
    )
    fuel_burn = parameters.initial_weight_newton * (1.0 - csdl.exp(exponent))
    fuel_burn.add_name("cruise_fuel_burn_newton")
    return fuel_burn


__all__ = ["FuelBurnParameters", "compute_fuel_burn"]
