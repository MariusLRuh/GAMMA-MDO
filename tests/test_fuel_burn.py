"""Analytic and derivative checks for the VortexAD-free fuel-burn model."""

import csdl_alpha as csdl
import numpy as np
import pytest

from gamma_mdo.core.boundary_surface_movement.fuel_burn import (
    FuelBurnParameters,
    compute_fuel_burn,
)


PARAMETERS = FuelBurnParameters(
    design_range_m=2.0e6,
    thrust_specific_fuel_consumption_kg_per_newton_second=1.7e-5,
    cruise_speed_m_s=230.0,
    initial_weight_newton=4.0e5,
)


def test_fuel_burn_matches_breguet_analytic_case():
    """Match the closed-form Breguet result for a known lift-to-drag ratio."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        fuel_burn = compute_fuel_burn(0.5, 0.025, PARAMETERS)
    finally:
        recorder.stop()

    lift_to_drag = 0.5 / 0.025
    exponent = (
        -PARAMETERS.design_range_m
        * PARAMETERS.gravity_m_s2
        * PARAMETERS.thrust_specific_fuel_consumption_kg_per_newton_second
        / (PARAMETERS.cruise_speed_m_s * lift_to_drag)
    )
    expected = PARAMETERS.initial_weight_newton * (1.0 - np.exp(exponent))
    assert float(np.asarray(fuel_burn.value).reshape(-1)[0]) == pytest.approx(
        expected, rel=1.0e-13
    )


def test_fuel_burn_lift_derivative_matches_centered_finite_difference():
    """Carry the analytic CSDL derivative through the complete fuel model."""
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    try:
        lift = csdl.Variable(name="test_lift_coefficient", value=np.array([0.5]))
        lift.set_as_design_variable(lower=0.2, upper=1.2)
        fuel_burn = compute_fuel_burn(lift, 0.025, PARAMETERS)
        fuel_burn.set_as_objective()
        analytic = csdl.derivative(fuel_burn, lift)
    finally:
        recorder.stop()

    analytic_value = float(np.asarray(analytic.value).reshape(-1)[0])
    simulator = csdl.experimental.JaxSimulator(recorder=recorder, gpu=False)
    baseline = np.asarray(lift.value).copy()
    errors = []
    for step in (1.0e-4, 1.0e-5, 1.0e-6):
        simulator[lift] = baseline + step
        simulator.run()
        plus = float(np.asarray(simulator[fuel_burn]).reshape(-1)[0])
        simulator[lift] = baseline - step
        simulator.run()
        minus = float(np.asarray(simulator[fuel_burn]).reshape(-1)[0])
        centered = (plus - minus) / (2.0 * step)
        errors.append(
            abs(analytic_value - centered)
            / max(abs(analytic_value), abs(centered), 1.0e-14)
        )
    simulator[lift] = baseline

    assert analytic_value < 0.0
    assert min(errors) < 1.0e-7


@pytest.mark.parametrize(
    "field",
    [
        "design_range_m",
        "thrust_specific_fuel_consumption_kg_per_newton_second",
        "cruise_speed_m_s",
        "initial_weight_newton",
        "gravity_m_s2",
    ],
)
def test_fuel_burn_parameters_reject_nonpositive_constants(field):
    """Reject nonphysical mission constants before graph construction."""
    values = {
        "design_range_m": 1.0,
        "thrust_specific_fuel_consumption_kg_per_newton_second": 1.0e-5,
        "cruise_speed_m_s": 1.0,
        "initial_weight_newton": 1.0,
        "gravity_m_s2": 1.0,
    }
    values[field] = 0.0
    with pytest.raises(ValueError, match=field):
        FuelBurnParameters(**values)
