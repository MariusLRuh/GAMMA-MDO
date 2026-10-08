from __future__ import annotations

from contextlib import contextmanager
import numpy as np
import pytest

import csdl_alpha as csdl

from bsm3.core.boundary_surface_movement.dafoam_csdl import (
    DAFoamAnalysisOperation,
    PYDAFoamBackend,
    add_csdl_inputs_to_da_options,
    build_local_volume_coordinate_map,
)
from bsm3.core.boundary_surface_movement.run_dafoam_gmsh import (
    FlowConfig,
    build_da_options,
    set_control_dict_max_iterations,
)


def test_flow_solver_controls_propagate_to_dafoam_options():
    config = FlowConfig(
        primal_min_res_tol=2.0e-10,
        primal_min_iterations=7,
        primal_max_iterations=4321,
        adjoint_gmres_relative_tolerance=3.0e-9,
        adjoint_gmres_absolute_tolerance=4.0e-15,
        adjoint_gmres_max_iterations=567,
        adjoint_gmres_restart=89,
    )

    options = build_da_options(config, ["aircraft"], ["farfield"])

    assert options["primalMinResTol"] == pytest.approx(2.0e-10)
    assert options["primalMinIters"] == 7
    assert options["adjEqnOption"]["gmresRelTol"] == pytest.approx(3.0e-9)
    assert options["adjEqnOption"]["gmresAbsTol"] == pytest.approx(4.0e-15)
    assert options["adjEqnOption"]["gmresMaxIters"] == 567
    assert options["adjEqnOption"]["gmresRestart"] == 89


def test_primal_min_res_tol_difference_propagates_to_dafoam_options():
    options = build_da_options(
        FlowConfig(primal_min_res_tol_difference=1.1),
        ["aircraft"],
        ["farfield"],
    )
    assert options["primalMinResTolDiff"] == pytest.approx(1.1)


def test_primal_min_res_tol_difference_defaults_to_dafoam_default():
    options = build_da_options(FlowConfig(), ["aircraft"], ["farfield"])
    assert options["primalMinResTolDiff"] == pytest.approx(1.0e2)


def test_primal_min_res_tol_difference_rejects_nonpositive_values():
    with pytest.raises(ValueError, match="primal_min_res_tol_difference"):
        FlowConfig(primal_min_res_tol_difference=0.0)
    with pytest.raises(ValueError, match="primal_min_res_tol_difference"):
        FlowConfig(primal_min_res_tol_difference=-1.0)


def test_build_da_options_converts_tuple_patches_to_native_lists():
    # DAFoam requires native lists; callers may pass tuples.
    config = FlowConfig()
    options = build_da_options(
        config,
        ("aircraft",),
        ("farfield",),
    )
    assert isinstance(options["designSurfaces"], list)
    assert isinstance(options["primalBC"]["U0"]["patches"], list)


def test_primal_max_iterations_updates_control_dict(tmp_path):
    system_directory = tmp_path / "system"
    system_directory.mkdir()
    control_dict = system_directory / "controlDict"
    control_dict.write_text(
        "// endTime 999;\n"
        "startTime       0;\n"
        "endTime         1000;\n",
        encoding="utf-8",
    )

    set_control_dict_max_iterations(tmp_path, 4321)

    assert control_dict.read_text(encoding="utf-8") == (
        "// endTime 999;\n"
        "startTime       0;\n"
        "endTime         4321;\n"
    )


def test_flow_solver_controls_reject_invalid_values():
    with pytest.raises(ValueError, match="primal_max_iterations"):
        FlowConfig(primal_min_iterations=10, primal_max_iterations=9)
    with pytest.raises(ValueError, match="relative_tolerance"):
        FlowConfig(adjoint_gmres_relative_tolerance=0.0)


class _AnalyticBackend:
    function_names = ("CL", "CD")

    def __init__(self):
        self.primal_calls = 0
        self.vjp_calls = 0

    def run_primal(self, inputs):
        self.primal_calls += 1
        coordinates = np.asarray(inputs["volume_coordinates"], dtype=float)
        patch_velocity = np.asarray(inputs["patch_velocity"], dtype=float)
        return {
            "CL": np.sum(coordinates**2)
            + 3.0 * patch_velocity[0]
            + patch_velocity[1] ** 2,
            "CD": patch_velocity[0] * np.sum(coordinates)
            + 0.5 * patch_velocity[1],
        }

    def compute_vjp(self, inputs, output_seeds):
        self.vjp_calls += 1
        coordinates = np.asarray(inputs["volume_coordinates"], dtype=float)
        patch_velocity = np.asarray(inputs["patch_velocity"], dtype=float)
        seed_cl = float(np.asarray(output_seeds["CL"]).reshape(-1)[0])
        seed_cd = float(np.asarray(output_seeds["CD"]).reshape(-1)[0])
        return {
            "volume_coordinates": (
                2.0 * seed_cl * coordinates
                + seed_cd * patch_velocity[0]
            ),
            "patch_velocity": np.array(
                [
                    3.0 * seed_cl + seed_cd * np.sum(coordinates),
                    2.0 * patch_velocity[1] * seed_cl + 0.5 * seed_cd,
                ]
            ),
        }


def test_dafoam_custom_operation_exposes_functions_and_vjp():
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    coordinates = csdl.Variable(
        name="volume_coordinates",
        value=np.arange(12, dtype=float).reshape((4, 3)) / 10.0,
    )
    patch_velocity = csdl.Variable(
        name="patch_velocity", value=np.array([200.0, 2.0])
    )
    backend = _AnalyticBackend()
    outputs = DAFoamAnalysisOperation(backend).evaluate(
        coordinates,
        patch_velocity=patch_velocity,
    )
    derivatives = csdl.derivative(
        [outputs["CL"], outputs["CD"]],
        [coordinates, patch_velocity],
    )
    recorder.stop()

    x = coordinates.value
    q = patch_velocity.value
    assert outputs["CL"].value[0] == pytest.approx(
        np.sum(x**2) + 3.0 * q[0] + q[1] ** 2
    )
    assert outputs["CD"].value[0] == pytest.approx(
        q[0] * np.sum(x) + 0.5 * q[1]
    )
    np.testing.assert_allclose(
        derivatives[outputs["CL"], coordinates].value.reshape(x.shape),
        2.0 * x,
    )
    np.testing.assert_allclose(
        derivatives[outputs["CD"], coordinates].value.reshape(x.shape),
        np.full_like(x, q[0]),
    )
    np.testing.assert_allclose(
        derivatives[outputs["CL"], patch_velocity].value.reshape(-1),
        np.array([3.0, 2.0 * q[1]]),
    )
    np.testing.assert_allclose(
        derivatives[outputs["CD"], patch_velocity].value.reshape(-1),
        np.array([np.sum(x), 0.5]),
    )
    assert backend.primal_calls == 1
    assert backend.vjp_calls >= 2


def test_dafoam_backend_contract_matches_central_finite_difference():
    backend = _AnalyticBackend()
    inputs = {
        "volume_coordinates": np.arange(9, dtype=float).reshape((3, 3)) / 7.0,
        "patch_velocity": np.array([180.0, -1.5]),
    }
    seeds = {"CL": np.array([0.7]), "CD": np.array([-1.2])}
    analytic = backend.compute_vjp(inputs, seeds)

    def objective(values):
        functions = backend.run_primal(values)
        return sum(
            float(np.asarray(functions[name]).reshape(-1)[0])
            * float(seed[0])
            for name, seed in seeds.items()
        )

    step = 1.0e-6
    for input_name, value in inputs.items():
        finite_difference = np.zeros_like(value)
        for index in np.ndindex(value.shape):
            plus = {name: array.copy() for name, array in inputs.items()}
            minus = {name: array.copy() for name, array in inputs.items()}
            plus[input_name][index] += step
            minus[input_name][index] -= step
            finite_difference[index] = (
                objective(plus) - objective(minus)
            ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic[input_name],
            finite_difference,
            rtol=2.0e-8,
            atol=2.0e-7,
        )


class _PointSolver:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float)

    def getOFMeshPoints(self, output):
        output[:] = self.points.reshape(-1)


class _PointDAFoam:
    def __init__(self, points):
        self.solver = _PointSolver(points)
        self._number = len(points)

    def getNLocalPoints(self):
        return self._number


def test_local_openfoam_points_map_back_to_global_gmsh_order():
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )
    local = reference[[3, 1, 3, 0]] + 1.0e-12
    mapping = build_local_volume_coordinate_map(
        _PointDAFoam(local), reference
    )
    np.testing.assert_array_equal(mapping, np.array([3, 1, 3, 0]))


def test_local_coordinate_mapping_rejects_a_different_mesh():
    reference = np.eye(3)
    with pytest.raises(ValueError, match="Could not match"):
        build_local_volume_coordinate_map(
            _PointDAFoam(np.array([[10.0, 10.0, 10.0]])),
            reference,
        )


def test_dafoam_input_option_builder_is_additive_and_nonmutating():
    original = {"solverName": "DARhoSimpleCFoam", "inputInfo": {"other": {}}}
    result = add_csdl_inputs_to_da_options(
        original,
        farfield_patches=("farfield", "inlet"),
    )
    assert original == {
        "solverName": "DARhoSimpleCFoam",
        "inputInfo": {"other": {}},
    }
    assert result["inputInfo"]["aero_vol_coords"] == {
        "type": "volCoord",
        "components": ["solver", "function"],
    }
    assert result["inputInfo"]["patch_velocity"]["patches"] == [
        "farfield",
        "inlet",
    ]


class _LinearAdjointSolverAD:
    def __init__(self, model):
        self.model = model

    def calcJacTVecProduct(
        self,
        input_name,
        input_type,
        jacobian_input,
        output_name,
        output_type,
        seed,
        product,
    ):
        del input_type, jacobian_input
        seed_array = np.asarray(seed, dtype=float)
        if output_type == "function":
            if input_name == "dafoam_solver_states":
                product[:] = self.model.state_gradient[output_name] * seed_array[0]
            elif input_name == "aero_vol_coords":
                product[:] = self.model.direct_volume[output_name] * seed_array[0]
            elif input_name == "patch_velocity":
                product[:] = self.model.direct_flow[output_name] * seed_array[0]
            else:
                raise KeyError(input_name)
        elif output_name == "aero_residuals" and output_type == "residual":
            if input_name == "aero_vol_coords":
                product[:] = -self.model.state_from_volume.T @ seed_array
            elif input_name == "patch_velocity":
                product[:] = -self.model.state_from_flow.T @ seed_array
            else:
                raise KeyError(input_name)
        else:
            raise ValueError((output_name, output_type))


class _LinearAdjointDAFoam:
    def __init__(self, model):
        self.model = model
        self.solverAD = _LinearAdjointSolverAD(model)
        self.comm = type("_Comm", (), {"size": 1})()
        self.inputs = None
        self.states = None

    def getOption(self, name):
        if name == "inputInfo":
            return {
                "aero_vol_coords": {
                    "type": "volCoord",
                    "components": ["solver", "function"],
                },
                "patch_velocity": {
                    "type": "patchVelocity",
                    "components": ["solver", "function"],
                },
            }
        raise KeyError(name)

    def set_solver_input(self, inputs):
        self.inputs = inputs

    def setStates(self, states):
        self.states = np.asarray(states, dtype=float).copy()


class _LinearAdjointModel:
    def __init__(self):
        self.state_from_volume = np.array(
            [
                [1.0, 2.0, 0.0, -1.0, 0.5, 0.0],
                [0.0, -0.5, 1.0, 0.0, 2.0, -1.0],
            ]
        )
        self.state_from_flow = np.array([[0.2, 1.0], [-0.3, 0.5]])
        self.state_gradient = {
            "CL": np.array([2.0, -1.0]),
            "CD": np.array([0.5, 3.0]),
        }
        self.direct_volume = {
            "CL": np.linspace(0.1, 0.6, 6),
            "CD": np.linspace(-0.3, 0.2, 6),
        }
        self.direct_flow = {
            "CL": np.array([0.7, -0.2]),
            "CD": np.array([-0.4, 0.9]),
        }


def test_set_deterministic_baseline_requires_converged_state():
    backend = object.__new__(PYDAFoamBackend)
    backend._cached_states = None
    with pytest.raises(RuntimeError, match="No converged state"):
        backend.set_deterministic_baseline()


def test_reset_primal_state_clears_cache():
    backend = object.__new__(PYDAFoamBackend)
    backend._cached_states = np.ones(3)
    backend._cached_inputs = {"x": np.ones(2)}
    backend.reset_primal_state()
    assert backend._cached_states is None
    assert backend._cached_inputs is None


def test_set_deterministic_baseline_freezes_cached_state():
    backend = object.__new__(PYDAFoamBackend)
    backend._cached_states = np.array([1.0, 2.0, 3.0])
    backend._baseline_states = None
    backend.set_deterministic_baseline()
    np.testing.assert_array_equal(backend._baseline_states, [1.0, 2.0, 3.0])
    # A later cache change must not mutate the frozen baseline.
    backend._cached_states[:] = 9.0
    np.testing.assert_array_equal(backend._baseline_states, [1.0, 2.0, 3.0])


def test_live_backend_total_vjp_has_correct_discrete_adjoint_sign():
    model = _LinearAdjointModel()
    dafoam = _LinearAdjointDAFoam(model)
    backend = object.__new__(PYDAFoamBackend)
    backend.dafoam_instance = dafoam
    backend.volume_input_name = "aero_vol_coords"
    backend.function_names = ("CL", "CD")
    backend.reference_volume_coordinates = np.zeros((2, 3))
    backend.local_to_global = np.array([0, 1])
    backend.volume_gradient_ownership = "replicated"

    global_coordinates = np.arange(6, dtype=float).reshape((2, 3)) / 10.0
    patch_velocity = np.array([200.0, 2.0])
    inputs = {
        "volume_coordinates": global_coordinates,
        "patch_velocity": patch_velocity,
    }
    local_coordinates = global_coordinates.reshape(-1)
    states = (
        model.state_from_volume @ local_coordinates
        + model.state_from_flow @ patch_velocity
    )
    backend._cached_inputs = {
        name: value.copy() for name, value in inputs.items()
    }
    backend._cached_states = states
    backend._cached_functions = {}
    backend._solve_adjoint = lambda right_hand_side: np.asarray(
        right_hand_side, dtype=float
    )

    @contextmanager
    def no_directory_change():
        yield

    backend._working_directory = no_directory_change

    seeds = {"CL": np.array([0.8]), "CD": np.array([-1.3])}
    derivatives = backend.compute_vjp(inputs, seeds)
    combined_state_gradient = (
        seeds["CL"][0] * model.state_gradient["CL"]
        + seeds["CD"][0] * model.state_gradient["CD"]
    )
    expected_volume = (
        seeds["CL"][0] * model.direct_volume["CL"]
        + seeds["CD"][0] * model.direct_volume["CD"]
        + model.state_from_volume.T @ combined_state_gradient
    ).reshape((2, 3))
    expected_flow = (
        seeds["CL"][0] * model.direct_flow["CL"]
        + seeds["CD"][0] * model.direct_flow["CD"]
        + model.state_from_flow.T @ combined_state_gradient
    )
    np.testing.assert_allclose(
        derivatives["volume_coordinates"], expected_volume
    )
    np.testing.assert_allclose(derivatives["patch_velocity"], expected_flow)


def test_flow_config_validates_nu_tilda():
    """Reject a non-finite or negative Spalart-Allmaras freestream value."""
    import dataclasses
    import math

    for bad in (math.nan, math.inf, -1.0e-6):
        with pytest.raises(ValueError, match="nu_tilda_m2_per_s"):
            dataclasses.replace(FlowConfig(), nu_tilda_m2_per_s=bad)

    assert dataclasses.replace(
        FlowConfig(), nu_tilda_m2_per_s=0.0
    ).nu_tilda_m2_per_s == 0.0


def test_build_da_options_applies_both_previously_dead_flow_fields():
    """Propagate nu_tilda and the wall-function flag into the DAFoam options."""
    import dataclasses

    farfield = ("inlet", "outlet", "farfield")
    config = dataclasses.replace(
        FlowConfig(), nu_tilda_m2_per_s=1.5e-4, use_wall_functions=True
    )
    options = build_da_options(config, ["aircraft"], list(farfield))
    primal_bc = options["primalBC"]

    assert primal_bc["nuTilda0"] == {
        "variable": "nuTilda",
        "patches": list(farfield),
        "value": [1.5e-4],
    }
    # DAFoam requires native containers, not tuples or NumPy arrays.
    assert type(primal_bc["nuTilda0"]["patches"]) is list
    assert type(primal_bc["nuTilda0"]["value"]) is list
    assert all(type(name) is str for name in primal_bc["nuTilda0"]["patches"])
    assert type(primal_bc["nuTilda0"]["value"][0]) is float

    assert primal_bc["useWallFunction"] is True

    resolved = build_da_options(
        dataclasses.replace(config, use_wall_functions=False),
        ["aircraft"],
        list(farfield),
    )
    assert resolved["primalBC"]["useWallFunction"] is False


def test_wall_function_cli_flag_defaults_to_the_dataclass_value():
    """Accept both flag forms and default to FlowConfig's own value."""
    from bsm3.core.boundary_surface_movement.run_dafoam_gmsh import make_parser

    parser = make_parser()
    required = ["--reuse-openfoam-mesh"]

    assert parser.parse_args(required).wall_functions is (
        FlowConfig().use_wall_functions
    )
    assert parser.parse_args(required + ["--wall-functions"]).wall_functions is True
    assert (
        parser.parse_args(required + ["--no-wall-functions"]).wall_functions
        is False
    )




def test_run_dafoam_gmsh_communicator_annotations_resolve_without_mpi():
    """Resolve the three communicator annotations with no mpi4py import.

    The annotations previously named a module-global ``MPI`` that did not
    exist, so ``typing.get_type_hints`` failed. They now use a structural
    protocol, which resolves in this environment where mpi4py is absent.
    """
    import typing

    from bsm3.core.boundary_surface_movement import run_dafoam_gmsh as module

    assert not hasattr(module, "MPI"), "mpi4py must not be imported eagerly"

    for function in (
        module.rank0_print,
        module.apply_custom_mesh_deformation,
        module.run_dafoam,
    ):
        hints = typing.get_type_hints(function)
        assert hints["comm"] is module._Communicator

    # A real-shaped communicator satisfies the protocol structurally.
    class _Comm:
        rank = 0
        size = 1

        def Barrier(self):
            return None

        def bcast(self, obj, root=0):
            return obj

    assert isinstance(_Comm(), module._Communicator)

    # The annotations carry no noqa suppression.
    import pathlib

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    assert "noqa: F821" not in source
