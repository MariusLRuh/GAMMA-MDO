# region Imports and Setup

import csdl_alpha as csdl
import numpy as np
import lsdo_function_spaces as lfs

from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
    VolumeSectionalParameterization,
    VolumeSectionalParameterizationInputs
)
from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

import lsdo_geo
from dataclasses import asdict, dataclass, fields
from typing import Union


M_TO_FT = 3.280839895013123


def _stack_coefficients_csdl(function_set: lfs.FunctionSet) -> csdl.Variable:
    reshaped_total_coeffs = []
    for function in function_set.functions.values():
        reshaped_total_coeffs.append(csdl.reshape(function.coefficients, (-1, 3)))
    return csdl.vstack(reshaped_total_coeffs)


def _stack_coefficients_numpy(function_set: lfs.FunctionSet) -> np.ndarray:
    reshaped_total_coeffs = []
    for function in function_set.functions.values():
        reshaped_total_coeffs.append(
            np.asarray(function.coefficients.value, dtype=float).reshape((-1, 3))
        )
    return np.vstack(reshaped_total_coeffs)


def _rotate_points_about_y_axis_csdl(
    points: csdl.Variable,
    angle_degrees: csdl.Variable,
    pivot_point: csdl.Variable,
) -> csdl.Variable:
    pivot_point = csdl.reshape(pivot_point, (1, 3))
    pivot_rows = csdl.matmat(np.ones((points.shape[0], 1), dtype=float), pivot_point)
    centered_points = points - pivot_rows

    x_coordinates = centered_points[csdl.slice[:, 0:1]]
    y_coordinates = centered_points[csdl.slice[:, 1:2]]
    z_coordinates = centered_points[csdl.slice[:, 2:3]]

    angle_radians = angle_degrees * (np.pi / 180.0)
    cosine = csdl.cos(angle_radians)
    sine = csdl.sin(angle_radians)

    rotated_x = cosine * x_coordinates + sine * z_coordinates
    rotated_z = -sine * x_coordinates + cosine * z_coordinates
    rotated_points = csdl.concatenate((rotated_x, y_coordinates, rotated_z), axis=1)
    return rotated_points + pivot_rows


def _smoothstep_numpy(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def _apply_stacked_coefficients_csdl(
    function_set: lfs.FunctionSet,
    stacked_coefficients: csdl.Variable,
) -> None:
    offset = 0
    for function in function_set.functions.values():
        coefficient_shape = function.coefficients.shape
        num_rows = int(np.prod(coefficient_shape) // 3)
        function.coefficients = stacked_coefficients[
            csdl.slice[offset : offset + num_rows, :]
        ].reshape(coefficient_shape)
        offset += num_rows

@dataclass(frozen=True)
class TransportCGEstimateInputsImperial:
    fuselage_nose_x_ft: Union[float, csdl.Variable]
    fuselage_tail_x_ft: Union[float, csdl.Variable]
    cabin_start_x_ft: Union[float, csdl.Variable]
    cabin_end_x_ft: Union[float, csdl.Variable]
    cargo_start_x_ft: Union[float, csdl.Variable]
    cargo_end_x_ft: Union[float, csdl.Variable]
    wing_root_le_x_ft: Union[float, csdl.Variable]
    wing_root_te_x_ft: Union[float, csdl.Variable]
    wing_yehudi_le_x_ft: Union[float, csdl.Variable]
    wing_yehudi_te_x_ft: Union[float, csdl.Variable]
    wing_tip_le_x_ft: Union[float, csdl.Variable]
    wing_tip_te_x_ft: Union[float, csdl.Variable]
    horizontal_tail_root_le_x_ft: Union[float, csdl.Variable]
    horizontal_tail_root_te_x_ft: Union[float, csdl.Variable]
    horizontal_tail_tip_le_x_ft: Union[float, csdl.Variable]
    horizontal_tail_tip_te_x_ft: Union[float, csdl.Variable]
    vertical_tail_root_le_x_ft: Union[float, csdl.Variable]
    vertical_tail_root_te_x_ft: Union[float, csdl.Variable]
    vertical_tail_tip_le_x_ft: Union[float, csdl.Variable]
    vertical_tail_tip_te_x_ft: Union[float, csdl.Variable]
    nacelle_inlet_x_ft: Union[float, csdl.Variable]
    nacelle_outlet_x_ft: Union[float, csdl.Variable]
    main_gear_x_ft: Union[float, csdl.Variable]
    nose_gear_x_ft: Union[float, csdl.Variable]
    wing_MAC: Union[float, csdl.Variable]
    fuselage_height: Union[float, csdl.Variable]
    fuselage_diameter: Union[float, csdl.Variable]
    wing_sweep_quarter_chord: Union[float, csdl.Variable]
    htail_sweep_quarter_chord: Union[float, csdl.Variable]
    tail_moment_arm: Union[float, csdl.Variable]


def evaluate_E175_geometry_parameterization(
    fuse_function_set: lfs.FunctionSet,
    wing_function_set: lfs.FunctionSet,
    htail_function_set: lfs.FunctionSet,
    wing_area_dv: csdl.Variable,
    wing_AR_dv: csdl.Variable,
    htail_area_dv: csdl.Variable,
    htail_AR_dv: csdl.Variable,
    wing_twist_dvs: csdl.Variable,
    wing_camber_dvs: csdl.Variable,
    wing_thickness_dvs: csdl.Variable,
    wing_translation_x: csdl.Variable,
    htail_root_rotation_degrees: csdl.Variable,
    fuselage_diameter_scale: csdl.Variable,
    num_ffd_coefficients_chordwise = 8,
    num_ffd_sections = 11,
) -> TransportCGEstimateInputsImperial:
    wing_geometry = lsdo_geo.Geometry(functions=wing_function_set.functions, space=wing_function_set.space)
    htail_geometry = lsdo_geo.Geometry(functions=htail_function_set.functions, space=htail_function_set.space)
    fuse_geometry = lsdo_geo.Geometry(functions=fuse_function_set.functions, space=fuse_function_set.space)

    # wing_geometry.plot()
    ####################### WING SECTION KEY POINTS #######################
    wing_LE_center = wing_geometry.project(np.array([10.882, 0., 0.2]), plot=False)
    wing_TE_center = wing_geometry.project(np.array([16.326, 0., -0.009]), plot=False)

    wing_LE_yehudi_right = wing_geometry.project(np.array([13.508, 4.863, 0.684]), plot=False)
    wing_LE_tip_right = wing_geometry.project(np.array([17.324, 11.911, 1.428]), plot=False)
    wing_TE_yehudi_right = wing_geometry.project(np.array([16.421, 4.863, 0.623]), plot=False)
    wing_TE_tip_right = wing_geometry.project(np.array([18.314, 11.911, 1.391]), plot=False)

    wing_LE_yehudi_left = wing_geometry.project(np.array([13.508, -4.863, 0.684]), plot=False)
    wing_LE_tip_left = wing_geometry.project(np.array([17.324, -11.911, 1.428]), plot=False)
    wing_TE_yehudi_left = wing_geometry.project(np.array([16.421, -4.863, 0.623]), plot=False)
    wing_TE_tip_left = wing_geometry.project(np.array([18.314, -11.911, 1.391]), plot=False)

    ####################### TAIL SECTION KEY POINTS #######################
    htail_LE_center = htail_geometry.project(np.array([26., 0., 1.572]), plot=False)
    htail_TE_center = htail_geometry.project(np.array([29.534, 0., 1.572]), plot=False)

    htail_LE_tip_right = htail_geometry.project(np.array([29.7, 5.0, 1.572]), plot=False)
    htail_TE_tip_right = htail_geometry.project(np.array([30.816, 5.0, 1.572]), plot=False)

    htail_LE_tip_left = htail_geometry.project(np.array([29.7, -5.0, 1.572]), plot=False)
    htail_TE_tip_left = htail_geometry.project(np.array([30.816, -5.0, 1.572]), plot=False)

    ####################### FUSELAGE SECTION KEY POINTS #######################
    fuse_tail_pt = fuse_geometry.project(np.array([31.680, 0., 1.901]), plot=False)
    fuse_nose_pt = fuse_geometry.project(np.array([0., 0., 0.]), plot=False)

    cabin_start_pt = fuse_geometry.project(np.array([6., 0., -0.5]), plot=False)
    cabin_end_pt = fuse_geometry.project(np.array([25.56, 0., 0.03]), plot=False)

    cargo_start_pt = fuse_geometry.project(np.array([10.0, 0., -0.5]), plot=False)
    cargo_end_pt = fuse_geometry.project(np.array([20.0, 0., -0.5]), plot=False)

    nose_landing_gear_pt = fuse_geometry.project(np.array([3.9, 0., -0.5]), plot=False)
    main_landing_gear_pt = fuse_geometry.project(np.array([16.25, 0., -0.6]), plot=False)

    fuselage_height_1 = fuse_geometry.project(np.array([18.025, 0., 2.625]), plot=False)
    fuselage_height_2 = fuse_geometry.project(np.array([18.025, 0., -0.725]), plot=False)

    fuselage_diameter_1 = fuse_geometry.project(np.array([18.025, -1.505, 0.950]), plot=False)
    fuselage_diameter_2 = fuse_geometry.project(np.array([18.025, 1.505, 0.950]), plot=False)


    num_semispan_ffd_sections = num_ffd_sections // 2 + 1
    # Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
    #       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
    ffd_block = construct_ffd_block_around_entities(
        entities=wing_geometry,
        num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
        degree=(3, 3, 1),
    )

    num_ffd_sections_htail = 3
    ffd_block_htail = construct_ffd_block_around_entities(
        entities=htail_geometry,
        num_coefficients=(2, num_ffd_sections_htail, 2),
        degree=(1, 1, 1),
    )

    ffd_sectional_parameterization = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization",
        parameterized_points=ffd_block.coefficients,
        principal_parametric_dimension=1,
    )

    ffd_sectional_parameterization_htail = VolumeSectionalParameterization(
        name="ffd_sectional_parameterization_htail",
        parameterized_points=ffd_block_htail.coefficients,
        principal_parametric_dimension=1,
    )

    space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(1,), coefficients_shape=(2,))
    space_of_cubic_8_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(3,), coefficients_shape=(8,))
    space_of_linear_3_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(1,), coefficients_shape=(3,))

    # Chord stretch is parameterized on the right semispan and mirrored so the solver
    # cannot create a left/right asymmetric chord distribution.
    chord_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
                                            coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='chord_stretching_b_spline_coefficients')

    # Span translation is parameterized with a smooth semispan B-spline and mirrored
    # antisymmetrically so the root stays on the symmetry plane.
    wingspan_stretching_b_spline = lfs.Function(
        space=space_of_cubic_8_dof_b_splines,
        coefficients=csdl.Variable(shape=(8,), value=np.array([0., 0., 0., 0., 0., 0., 0., 0.])),
        name='wingspan_stretching_b_spline_coefficients',
    )

    wing_sweep_b_spline = lfs.Function(
        space=space_of_linear_3_dof_b_splines,
        coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
        name='wing_sweep_b_spline_coefficients',
    )

    tail_span_stretching_b_spline = lfs.Function(
        space=space_of_linear_3_dof_b_splines,
        coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
        name='tail_span_stretching_b_spline_coefficients',
    )

    tail_chord_stretching_b_spline = lfs.Function(
        space=space_of_linear_2_dof_b_splines,
        coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])),
        name='tail_chord_stretching_b_spline_coefficients',
    )

    tail_sweep_b_spline = lfs.Function(
        space=space_of_linear_3_dof_b_splines,
        coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
        name='tail_sweep_b_spline_coefficients',
    )

    # The full-wing FFD sections run from left tip -> root -> right tip, so use a semispan
    # twist profile (root -> tip) and mirror it to get a symmetric nonlinear distribution.
    twist_b_spline = lfs.Function(
        space=space_of_cubic_8_dof_b_splines,
        coefficients=wing_twist_dvs, # value=np.linspace(10, -10, 8) * np.pi / 180.0,
        name='twist_b_spline',
    )


    semispan_parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_semispan_ffd_sections).reshape((-1, 1))
    right_semispan_chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs)
    chord_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    chord_stretch_sectional_parameters = chord_stretch_sectional_parameters.set(
        csdl.slice[num_ffd_sections//2:],
        right_semispan_chord_stretch_sectional_parameters,
    )
    chord_stretch_sectional_parameters = chord_stretch_sectional_parameters.set(
        csdl.slice[:num_ffd_sections//2],
        right_semispan_chord_stretch_sectional_parameters[:0:-1],
    )

    wingspan_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    right_semispan_wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs)
    wingspan_stretch_sectional_parameters = wingspan_stretch_sectional_parameters.set(
        csdl.slice[num_ffd_sections//2:],
        right_semispan_wingspan_stretch_sectional_parameters,
    )
    wingspan_stretch_sectional_parameters = wingspan_stretch_sectional_parameters.set(
        csdl.slice[:num_ffd_sections//2],
        -right_semispan_wingspan_stretch_sectional_parameters[:0:-1],
    )

    right_semispan_wing_sweep_translation_sectional_parameters = wing_sweep_b_spline.evaluate(
        semispan_parametric_b_spline_inputs
    )
    wing_sweep_translation_sectional_parameters = csdl.Variable(
        shape=(num_ffd_sections,),
        value=0.,
    )
    wing_sweep_translation_sectional_parameters = (
        wing_sweep_translation_sectional_parameters.set(
            csdl.slice[num_ffd_sections//2:],
            right_semispan_wing_sweep_translation_sectional_parameters,
        )
    )
    wing_sweep_translation_sectional_parameters = (
        wing_sweep_translation_sectional_parameters.set(
            csdl.slice[:num_ffd_sections//2],
            right_semispan_wing_sweep_translation_sectional_parameters[:0:-1],
        )
    )

    right_semispan_twist_sectional_parameters = twist_b_spline.evaluate(semispan_parametric_b_spline_inputs)
    wing_twist_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
    wing_twist_sectional_parameters = wing_twist_sectional_parameters.set(
        csdl.slice[num_ffd_sections//2:],
        right_semispan_twist_sectional_parameters,
    )
    wing_twist_sectional_parameters = wing_twist_sectional_parameters.set(
        csdl.slice[:num_ffd_sections//2],
        right_semispan_twist_sectional_parameters[:0:-1],
    )

    # tail 
    semispan_parametric_b_spline_inputs_tail = np.linspace(0.0, 1.0, 2).reshape((-1, 1))

    right_semispan_tail_chord_stretch_sectional_parameters = tail_chord_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
    chord_stretch_sectional_parameters_tail = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
    chord_stretch_sectional_parameters_tail = chord_stretch_sectional_parameters_tail.set(
        csdl.slice[1:],
        right_semispan_tail_chord_stretch_sectional_parameters,
    )
    chord_stretch_sectional_parameters_tail = chord_stretch_sectional_parameters_tail.set(
        csdl.slice[:1],
        right_semispan_tail_chord_stretch_sectional_parameters[:0:-1],
    )

    right_semispan_tail_span_stretch_sectional_parameters = tail_span_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
    htail_span_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
    htail_span_stretch_sectional_parameters = htail_span_stretch_sectional_parameters.set(
        csdl.slice[1:],
        right_semispan_tail_span_stretch_sectional_parameters,
    )
    htail_span_stretch_sectional_parameters = htail_span_stretch_sectional_parameters.set(
        csdl.slice[:1],
        -right_semispan_tail_span_stretch_sectional_parameters[:0:-1],
    )

    right_semispan_tail_sweep_translation_sectional_parameters = tail_sweep_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
    tail_sweep_translation_sectional_parameters = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
    tail_sweep_translation_sectional_parameters = tail_sweep_translation_sectional_parameters.set(
        csdl.slice[1:],
        right_semispan_tail_sweep_translation_sectional_parameters,
    )
    tail_sweep_translation_sectional_parameters = tail_sweep_translation_sectional_parameters.set(
        csdl.slice[:1],
        right_semispan_tail_sweep_translation_sectional_parameters[:0:-1],
    )

    # Evaluate the sectional parameterization to get the FFD coefficients
    sectional_parameters = VolumeSectionalParameterizationInputs()
    sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
    sectional_parameters.add_sectional_translation(axis=0, translation=wing_sweep_translation_sectional_parameters)
    sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
    sectional_parameters.add_sectional_rotation(axis=1, rotation=wing_twist_sectional_parameters)

    sectional_parameters_htail = VolumeSectionalParameterizationInputs()
    sectional_parameters_htail.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters_tail)
    sectional_parameters_htail.add_sectional_translation(axis=1, translation=htail_span_stretch_sectional_parameters)
    sectional_parameters_htail.add_sectional_translation(axis=0, translation=tail_sweep_translation_sectional_parameters)


    ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)
    ffd_coefficients_htail = ffd_sectional_parameterization_htail.evaluate(sectional_parameters_htail, plot=False)

    # Parameterize each spanwise FFD section as a rectangular airfoil in the local
    # thickness direction "w". Interior chordwise rows and non-winglet spanwise
    # sections get percent camber and percent thickness changes based on the local
    # section chord length.
    delta_camber_percent = csdl.Variable(
        shape=(num_ffd_coefficients_chordwise, num_ffd_sections),
        value=0.,
    )
    delta_thickness_percent = csdl.Variable(
        shape=(num_ffd_coefficients_chordwise, num_ffd_sections),
        value=0.,
    )

    # Example initialization for debugging:
    # base_delta_camber_percent_design_dof_values = np.array([
    #     [0.0, 0.1, 0.2, 0.3],
    #     [0.0, 0.2, 0.3, 0.4],
    #     [0.0, 0.3, 0.5, 0.6],
    #     [0.0, 0.3, 0.5, 0.6],
    #     [0.0, 0.2, 0.3, 0.4],
    #     [0.0, 0.1, 0.2, 0.3],
    # ])
    # delta_camber_percent_design_dof_values = np.vstack([
    #     np.interp(
    #         np.linspace(0.0, 1.0, num_semispan_ffd_sections - 1),
    #         np.linspace(0.0, 1.0, base_delta_camber_percent_design_dof_values.shape[1]),
    #         base_delta_camber_percent_design_dof_values[row_index],
    #     )
    #     for row_index in range(base_delta_camber_percent_design_dof_values.shape[0])
    # ])
    # delta_camber_percent_design_dof = csdl.Variable(
    #     shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
    #     # value=delta_camber_percent_design_dof_values,
    #     # value=np.zeros((num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1)),
    #     value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 10,
    #     name='delta_camber_percent_design_dof',
    # )
    # delta_thickness_percent_design_dof = csdl.Variable(
    #     shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
    #     # value=np.zeros((num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1)),
    #     value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 5,
    #     name='delta_thickness_percent_design_dof',
    # )
    # delta_camber_percent_design_dof.set_as_design_variable(lower=-10.0, upper=10.0, scaler=0.1)
    # delta_thickness_percent_design_dof.set_as_design_variable(lower=-5.0, upper=5.0, scaler=0.2)

    delta_camber_percent = delta_camber_percent.set(
        csdl.slice[1:-1, num_ffd_sections//2:-1],
        wing_camber_dvs,
    )
    delta_camber_percent = delta_camber_percent.set(
        csdl.slice[1:-1, 1:num_ffd_sections//2],
        wing_camber_dvs[:, 1:][:, ::-1],
    )
    delta_thickness_percent = delta_thickness_percent.set(
        csdl.slice[1:-1, num_ffd_sections//2:-1],
        wing_thickness_dvs,
    )
    delta_thickness_percent = delta_thickness_percent.set(
        csdl.slice[1:-1, 1:num_ffd_sections//2],
        wing_thickness_dvs[:, 1:][:, ::-1],
    )

    local_w_vectors = ffd_coefficients[:, :, 1, :] - ffd_coefficients[:, :, 0, :]
    local_w_gap = csdl.norm(local_w_vectors, axes=(2,))
    local_w_hat = local_w_vectors / csdl.expand(local_w_gap, local_w_vectors.shape, 'ij->ija')

    sectional_chord_length = 0.5 * (
        csdl.norm(
            ffd_coefficients[-1, :, 0, :] - ffd_coefficients[0, :, 0, :],
            axes=(1,),
        )
        + csdl.norm(
            ffd_coefficients[-1, :, 1, :] - ffd_coefficients[0, :, 1, :],
            axes=(1,),
        )
    )
    sectional_chord_length = csdl.expand(
        sectional_chord_length,
        (num_ffd_coefficients_chordwise, num_ffd_sections),
        'j->ij',
    )

    delta_camber = (delta_camber_percent / 100.0) * sectional_chord_length
    delta_thickness = (delta_thickness_percent / 100.0) * sectional_chord_length

    top_w_displacement = delta_camber + 0.5 * delta_thickness
    bottom_w_displacement = delta_camber - 0.5 * delta_thickness

    top_vector_displacement = csdl.expand(
        top_w_displacement, local_w_vectors.shape, 'ij->ija'
    ) * local_w_hat
    bottom_vector_displacement = csdl.expand(
        bottom_w_displacement, local_w_vectors.shape, 'ij->ija'
    ) * local_w_hat

    ffd_coefficients = ffd_coefficients.set(
        csdl.slice[:, :, 1, :],
        ffd_coefficients[:, :, 1, :] + top_vector_displacement,
    )
    ffd_coefficients = ffd_coefficients.set(
        csdl.slice[:, :, 0, :],
        ffd_coefficients[:, :, 0, :] + bottom_vector_displacement,
    )

    updated_local_w_gap = csdl.norm(
        ffd_coefficients[:, :, 1, :] - ffd_coefficients[:, :, 0, :],
        axes=(2,),
    )
    minimum_ffd_layer_gap = csdl.minimum(updated_local_w_gap)

    # Evaluate the FFD and set the coefficients of the geometry
    wing_geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
    wing_geometry.set_coefficients(wing_geometry_coefficients) 
    ffd_block.plot()

    htail_geometry_coefficients = ffd_block_htail.evaluate_ffd(coefficients=ffd_coefficients_htail, plot=False)
    htail_geometry.set_coefficients(htail_geometry_coefficients) 
    ffd_block_htail.plot()

    # wing_geometry.plot()
    # exit()

    # Wing area reference area computation
    # One side of the wing area is estimated as the area of two trapezoids: 
    #  - one with bases defined by the root chord and the yehudi chord, 
    #  - one with bases defined by the yehudi chord and the tip chord. 
    # The total area is then twice this area to account for both sides of the wing.

    # Trapz area for center to yehudi section, right semispan
    trapz_base_wing_center = csdl.norm(wing_geometry.evaluate(wing_LE_center) - wing_geometry.evaluate(wing_TE_center)) 
    trapz_base_wing_yehudi = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_right) - wing_geometry.evaluate(wing_TE_yehudi_right)) 
    # trapz height is the spanwise distance between the center and yehudi sections
    trapz_height_wing_yehudi = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_right)[1] - wing_geometry.evaluate(wing_LE_center)[1]) 
    trapz_area_wing_yehudi = 0.5 * (trapz_base_wing_center + trapz_base_wing_yehudi) * trapz_height_wing_yehudi

    # Trapz area for yehudi to tip section
    trapz_base_wing_tip = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_TE_tip_right)) 
    # trapz height is the spanwise distance between the yehudi and tip sections
    trapz_height_wing_tip = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right)[1] - wing_geometry.evaluate(wing_LE_yehudi_right)[1]) 
    trapz_area_wing_tip = 0.5 * (trapz_base_wing_yehudi + trapz_base_wing_tip) * trapz_height_wing_tip

    # Left semispan area is computed explicitly so the area metric remains meaningful
    # even if some future parameterization accidentally introduces asymmetry.
    trapz_base_wing_yehudi_left = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_left) - wing_geometry.evaluate(wing_TE_yehudi_left))
    trapz_height_wing_yehudi_left = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_left)[1] - wing_geometry.evaluate(wing_LE_center)[1])
    trapz_area_wing_yehudi_left = 0.5 * (trapz_base_wing_center + trapz_base_wing_yehudi_left) * trapz_height_wing_yehudi_left

    trapz_base_wing_tip_left = csdl.norm(wing_geometry.evaluate(wing_LE_tip_left) - wing_geometry.evaluate(wing_TE_tip_left))
    trapz_height_wing_tip_left = csdl.norm(wing_geometry.evaluate(wing_LE_tip_left)[1] - wing_geometry.evaluate(wing_LE_yehudi_left)[1])
    trapz_area_wing_tip_left = 0.5 * (trapz_base_wing_yehudi_left + trapz_base_wing_tip_left) * trapz_height_wing_tip_left

    estimated_wing_area = trapz_area_wing_yehudi + trapz_area_wing_tip + trapz_area_wing_yehudi_left + trapz_area_wing_tip_left
    print("Estimated Wing Area: ", estimated_wing_area.value)

    wing_span = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_LE_tip_left))
    wing_AR = wing_span**2 / estimated_wing_area
    print("Estimated Wing Aspect Ratio: ", wing_AR.value)


    # fuselage-htail connection constraints
    fuse_htail_connection_constraint = csdl.norm(fuse_geometry.evaluate(fuse_tail_pt) - htail_geometry.evaluate(htail_TE_center))

    # htail area reference area computation
    htail_trapz_base_htail_center = csdl.norm(htail_geometry.evaluate(htail_LE_center) - htail_geometry.evaluate(htail_TE_center))
    htail_trapz_base_htail_tip_right = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right) - htail_geometry.evaluate(htail_TE_tip_right))
    htail_trapz_height_htail_tip_right = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right)[1] - htail_geometry.evaluate(htail_LE_center)[1])
    htail_trapz_area_htail_tip_right = 0.5 * (htail_trapz_base_htail_center + htail_trapz_base_htail_tip_right) * htail_trapz_height_htail_tip_right

    htail_trapz_base_htail_tip_left = csdl.norm(htail_geometry.evaluate(htail_LE_tip_left) - htail_geometry.evaluate(htail_TE_tip_left))
    htail_trapz_height_htail_tip_left = csdl.norm(htail_geometry.evaluate(htail_LE_tip_left)[1] - htail_geometry.evaluate(htail_LE_center)[1])
    htail_trapz_area_htail_tip_left = 0.5 * (htail_trapz_base_htail_center + htail_trapz_base_htail_tip_left) * htail_trapz_height_htail_tip_left

    htail_area = htail_trapz_area_htail_tip_right + htail_trapz_area_htail_tip_left
    print("Estimated Htail Area: ", htail_area.value)

    htail_span = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right) - htail_geometry.evaluate(htail_LE_tip_left))
    htail_AR = htail_span**2 / htail_area
    print("Estimated Htail Aspect Ratio: ", htail_AR.value)

    geometry_solver = ParameterizationSolver()

    # Define the states for the parameterization solver (solver will manipulate these to
    # achieve the variables). The chord and span states are already mirrored, so the
    # inner solve cannot create a left/right asymmetric wing.
    geometry_solver.add_state(chord_stretching_b_spline.coefficients)
    geometry_solver.add_state(wingspan_stretching_b_spline.coefficients)
    geometry_solver.add_state(wing_sweep_b_spline.coefficients)
    geometry_solver.add_state(tail_span_stretching_b_spline.coefficients)
    geometry_solver.add_state(tail_chord_stretching_b_spline.coefficients)
    geometry_solver.add_state(tail_sweep_b_spline.coefficients)

    # wing_area_dv = csdl.Variable(shape=(1,), name='wing_area', value=np.array([70.]))
    # wing_area_dv.set_as_design_variable(lower=0.75 * 70., upper=1.25 * 70., scaler=0.02)
    # wing_AR_dv = csdl.Variable(shape=(1,), name='wing_AR', value=np.array([8.4]))
    # wing_AR_dv.set_as_design_variable(lower=0.75 * 8.4, upper=1.25 * 8.4, scaler=0.15)

    # htail_area_dv = csdl.Variable(shape=(1,), name='htail_area', value=np.array([23.25]))
    # htail_area_dv.set_as_design_variable(lower=0.75 * 23.25, upper=1.25 * 23.25, scaler=0.05)
    # htail_AR_dv = csdl.Variable(shape=(1,), name='htail_AR', value=np.array([4.3]))
    # htail_AR_dv.set_as_design_variable(lower=0.75 * 4.3, upper=1.25 * 4.3, scaler=0.3)

    geometric_variables = GeometricVariables()
    geometric_variables.add_variable(estimated_wing_area, wing_area_dv, penalty_value=None)
    geometric_variables.add_variable(wing_AR, wing_AR_dv, penalty_value=None)
    geometric_variables.add_variable(fuse_htail_connection_constraint, fuse_htail_connection_constraint.value, penalty_value=None)
    geometric_variables.add_variable(htail_area, htail_area_dv, penalty_value=None)
    geometric_variables.add_variable(htail_AR, htail_AR_dv, penalty_value=None)

    geometry_solver.evaluate(geometric_variables)

    fuselage_diameter_taper_before_empennage = True
    fuselage_diameter_taper_margin_htail_chords = 0.50
    fuselage_diameter_taper_length_htail_chords = 2.00

    # Apply the rigid-body/component-level post-solve motions here so the
    # downstream movement code consumes the already-parameterized geometry.
    wing_coefficients_csdl = _stack_coefficients_csdl(wing_function_set)
    wing_translation_vector = csdl.reshape(
        csdl.concatenate(
            (wing_translation_x, np.zeros((1,), dtype=float), np.zeros((1,), dtype=float)),
            axis=0,
        ),
        (1, 3),
    )
    wing_translation_rows = csdl.matmat(
        np.ones((wing_coefficients_csdl.shape[0], 1), dtype=float),
        wing_translation_vector,
    )
    _apply_stacked_coefficients_csdl(
        wing_function_set,
        wing_coefficients_csdl + wing_translation_rows,
    )

    htail_root_leading_point = htail_geometry.evaluate(htail_LE_center)
    htail_root_trailing_point = htail_geometry.evaluate(htail_TE_center)
    htail_root_quarter_chord = htail_root_leading_point + 0.25 * (
        htail_root_trailing_point - htail_root_leading_point
    )
    htail_coefficients_csdl = _stack_coefficients_csdl(htail_function_set)
    rotated_htail_coefficients = _rotate_points_about_y_axis_csdl(
        htail_coefficients_csdl,
        htail_root_rotation_degrees,
        htail_root_quarter_chord,
    )
    _apply_stacked_coefficients_csdl(
        htail_function_set,
        rotated_htail_coefficients,
    )

    current_fuse_coefficients = _stack_coefficients_numpy(fuse_function_set)
    current_htail_root_leading_point = np.asarray(
        htail_root_leading_point.value,
        dtype=float,
    ).reshape((1, 3))
    current_htail_root_trailing_point = np.asarray(
        htail_root_trailing_point.value,
        dtype=float,
    ).reshape((1, 3))
    htail_root_chord_proxy = max(
        float(current_htail_root_trailing_point[0, 0] - current_htail_root_leading_point[0, 0]),
        1e-12,
    )
    fuselage_centerline_point = np.array(
        [
            0.0,
            0.5
            * (
                float(np.min(current_fuse_coefficients[:, 1]))
                + float(np.max(current_fuse_coefficients[:, 1]))
            ),
            0.5
            * (
                float(np.min(current_fuse_coefficients[:, 2]))
                + float(np.max(current_fuse_coefficients[:, 2]))
            ),
        ],
        dtype=float,
    )
    fuselage_centerline_rows = np.broadcast_to(
        fuselage_centerline_point,
        current_fuse_coefficients.shape,
    ).copy()
    fuselage_radial_mask = np.broadcast_to(
        np.array([0.0, 1.0, 1.0], dtype=float),
        current_fuse_coefficients.shape,
    ).copy()
    fuselage_diameter_taper_x_end = min(
        float(current_htail_root_leading_point[0, 0]),
        float(current_htail_root_trailing_point[0, 0]),
    ) - fuselage_diameter_taper_margin_htail_chords * htail_root_chord_proxy
    fuselage_diameter_taper_length = (
        fuselage_diameter_taper_length_htail_chords * htail_root_chord_proxy
    )
    if fuselage_diameter_taper_before_empennage:
        fuselage_diameter_taper_parameter = (
            fuselage_diameter_taper_x_end - current_fuse_coefficients[:, 0]
        ) / max(fuselage_diameter_taper_length, 1e-12)
        fuselage_diameter_taper_weights = _smoothstep_numpy(
            fuselage_diameter_taper_parameter
        )
    else:
        fuselage_diameter_taper_weights = np.ones(
            current_fuse_coefficients.shape[0],
            dtype=float,
        )
    fuselage_diameter_taper_rows = np.broadcast_to(
        fuselage_diameter_taper_weights.reshape((-1, 1)),
        current_fuse_coefficients.shape,
    ).copy()
    fuse_coefficients_csdl = _stack_coefficients_csdl(fuse_function_set)
    fuselage_radial_offsets_csdl = (
        fuse_coefficients_csdl - fuselage_centerline_rows
    ) * fuselage_radial_mask * fuselage_diameter_taper_rows
    moved_fuse_coefficients = fuse_coefficients_csdl + (
        fuselage_diameter_scale - 1.0
    ) * fuselage_radial_offsets_csdl
    _apply_stacked_coefficients_csdl(
        fuse_function_set,
        moved_fuse_coefficients,
    )


    print("wing_area", estimated_wing_area.value)
    print("wing_AR", wing_AR.value)
    print("wing_twist_sectional_parameters [deg]", wing_twist_sectional_parameters.value * 180.0 / np.pi)
    print("chord_stretch_sectional_parameters", chord_stretch_sectional_parameters.value)
    print("wing_sweep_translation_sectional_parameters", wing_sweep_translation_sectional_parameters.value)
    print("wingspan_stretch_sectional_parameters", wingspan_stretch_sectional_parameters.value)
    print("minimum_ffd_layer_gap", minimum_ffd_layer_gap.value)

    print("htail_area", htail_area.value)
    print("htail_AR", htail_AR.value)


    # return cg_estimate_inputs

    wing_root_chord = csdl.norm(wing_geometry.evaluate(wing_LE_center) - wing_geometry.evaluate(wing_TE_center))
    wing_tip_chord = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_TE_tip_right))
    taper_ratio = wing_tip_chord / wing_root_chord

    MAC = (2/3) * wing_root_chord * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
    print("Estimated Wing MAC: ", MAC.value)

    fuselage_height = csdl.norm(fuse_geometry.evaluate(fuselage_height_1) - fuse_geometry.evaluate(fuselage_height_2))
    fuselage_diameter = csdl.norm(fuse_geometry.evaluate(fuselage_diameter_1) - fuse_geometry.evaluate(fuselage_diameter_2))
    print("Estimated Fuselage Height: ", fuselage_height.value)
    print("Estimated Fuselage Diameter: ", fuselage_diameter.value)

    # compute quarter chord sweep for wing and tail using the leading and trailing edge points at the center and yehudi sections
    wing_LE_center_point = wing_geometry.evaluate(wing_LE_center)
    wing_TE_center_point = wing_geometry.evaluate(wing_TE_center)
    wing_LE_yehudi_right_point = wing_geometry.evaluate(wing_LE_yehudi_right)
    wing_TE_yehudi_right_point = wing_geometry.evaluate(wing_TE_yehudi_right)
    wing_quarter_chord_center_point = wing_LE_center_point + 0.25 * (wing_TE_center_point - wing_LE_center_point)
    wing_quarter_chord_yehudi_right_point = wing_LE_yehudi_right_point + 0.25 * (wing_TE_yehudi_right_point - wing_LE_yehudi_right_point)
    wing_quarter_chord_sweep = csdl.arctan2(
        wing_quarter_chord_yehudi_right_point[0] - wing_quarter_chord_center_point[0],
        wing_quarter_chord_yehudi_right_point[1] - wing_quarter_chord_center_point[1],
    )

    htail_LE_center_point = htail_geometry.evaluate(htail_LE_center)
    htail_TE_center_point = htail_geometry.evaluate(htail_TE_center)
    htail_LE_yehudi_right_point = htail_geometry.evaluate(htail_LE_tip_right)
    htail_TE_yehudi_right_point = htail_geometry.evaluate(htail_TE_tip_right)
    htail_quarter_chord_center_point = htail_LE_center_point + 0.25 * (htail_TE_center_point - htail_LE_center_point)
    htail_quarter_chord_yehudi_right_point = htail_LE_yehudi_right_point + 0.25 * (htail_TE_yehudi_right_point - htail_LE_yehudi_right_point)
    htail_quarter_chord_sweep = csdl.arctan2(
        htail_quarter_chord_yehudi_right_point[0] - htail_quarter_chord_center_point[0],
        htail_quarter_chord_yehudi_right_point[1] - htail_quarter_chord_center_point[1],
    )

    # computing tail moment arm as the distance from the quarter chord point of the wing root to the quarter chord point of the htail root
    tail_moment_arm = csdl.norm(
        htail_quarter_chord_center_point - wing_quarter_chord_center_point
    )
    print("Estimated Tail Moment Arm: ", tail_moment_arm.value)

    print("Estimated Wing Quarter Chord Sweep: ", wing_quarter_chord_sweep.value * 180.0 / np.pi)
    print("Estimated Htail Quarter Chord Sweep: ", htail_quarter_chord_sweep.value * 180.0 / np.pi)

    cg_estimate_inputs = TransportCGEstimateInputsImperial(
        fuselage_nose_x_ft=fuse_geometry.evaluate(fuse_nose_pt)[0] * M_TO_FT,
        fuselage_tail_x_ft=fuse_geometry.evaluate(fuse_tail_pt)[0] * M_TO_FT,
        cabin_start_x_ft=fuse_geometry.evaluate(cabin_start_pt)[0] * M_TO_FT,
        cabin_end_x_ft=fuse_geometry.evaluate(cabin_end_pt)[0] * M_TO_FT,
        cargo_start_x_ft=fuse_geometry.evaluate(cargo_start_pt)[0] * M_TO_FT,
        cargo_end_x_ft=fuse_geometry.evaluate(cargo_end_pt)[0] * M_TO_FT,
        wing_root_le_x_ft=wing_geometry.evaluate(wing_LE_center)[0] * M_TO_FT,
        wing_root_te_x_ft=wing_geometry.evaluate(wing_TE_center)[0] * M_TO_FT,
        wing_yehudi_le_x_ft=wing_geometry.evaluate(wing_LE_yehudi_right)[0] * M_TO_FT,
        wing_yehudi_te_x_ft=wing_geometry.evaluate(wing_TE_yehudi_right)[0] * M_TO_FT,
        wing_tip_le_x_ft=wing_geometry.evaluate(wing_LE_tip_right)[0] * M_TO_FT,
        wing_tip_te_x_ft=wing_geometry.evaluate(wing_TE_tip_right)[0] * M_TO_FT,
        horizontal_tail_root_le_x_ft=htail_geometry.evaluate(htail_LE_center)[0] * M_TO_FT,
        horizontal_tail_root_te_x_ft=htail_geometry.evaluate(htail_TE_center)[0] * M_TO_FT,
        horizontal_tail_tip_le_x_ft=htail_geometry.evaluate(htail_LE_tip_right)[0] * M_TO_FT,
        horizontal_tail_tip_te_x_ft=htail_geometry.evaluate(htail_TE_tip_right)[0] * M_TO_FT,
        vertical_tail_root_le_x_ft=26.1 * M_TO_FT,
        vertical_tail_root_te_x_ft=31.2 * M_TO_FT,
        vertical_tail_tip_le_x_ft=29.3 * M_TO_FT,
        vertical_tail_tip_te_x_ft=31.0 * M_TO_FT,
        nacelle_inlet_x_ft=(wing_geometry.evaluate(wing_LE_yehudi_right)[0] - 2.9) * M_TO_FT,
        nacelle_outlet_x_ft=(wing_geometry.evaluate(wing_TE_yehudi_right)[0] - 3.1) * M_TO_FT,
        main_gear_x_ft=fuse_geometry.evaluate(main_landing_gear_pt)[0] * M_TO_FT,
        nose_gear_x_ft=fuse_geometry.evaluate(nose_landing_gear_pt)[0] * M_TO_FT,
        wing_MAC=MAC,
        wing_sweep_quarter_chord=wing_quarter_chord_sweep,
        htail_sweep_quarter_chord=htail_quarter_chord_sweep,
        fuselage_diameter=fuselage_diameter * M_TO_FT,
        fuselage_height=fuselage_height * M_TO_FT,
        tail_moment_arm=tail_moment_arm,
    )

    # print("fuselage_diameter_ft", cg_estimate_inputs.fuselage_diameter.value)
    # print("fuselage_height_ft", cg_estimate_inputs.fuselage_height.value)
    # exit("hi")

    return cg_estimate_inputs
    

    ######################### END EMBRAER 175 EXAMPLE CODE #########################






# # region Imports and Setup

# import csdl_alpha as csdl
# import numpy as np
# import lsdo_function_spaces as lfs

# from lsdo_geo.core.parameterization.free_form_deformation_functions import construct_ffd_block_around_entities
# from lsdo_geo.core.parameterization.volume_sectional_parameterization import (
#     VolumeSectionalParameterization,
#     VolumeSectionalParameterizationInputs
# )
# from lsdo_geo.core.parameterization.parameterization_solver import ParameterizationSolver, GeometricVariables

# import lsdo_geo
# from dataclasses import asdict, dataclass, fields
# from typing import Union


# M_TO_FT = 3.280839895013123


# def _stack_coefficients_csdl(function_set: lfs.FunctionSet) -> csdl.Variable:
#     reshaped_total_coeffs = []
#     for function in function_set.functions.values():
#         reshaped_total_coeffs.append(csdl.reshape(function.coefficients, (-1, 3)))
#     return csdl.vstack(reshaped_total_coeffs)


# def _stack_coefficients_numpy(function_set: lfs.FunctionSet) -> np.ndarray:
#     reshaped_total_coeffs = []
#     for function in function_set.functions.values():
#         reshaped_total_coeffs.append(
#             np.asarray(function.coefficients.value, dtype=float).reshape((-1, 3))
#         )
#     return np.vstack(reshaped_total_coeffs)


# def _rotate_points_about_y_axis_csdl(
#     points: csdl.Variable,
#     angle_degrees: csdl.Variable,
#     pivot_point: csdl.Variable,
# ) -> csdl.Variable:
#     pivot_point = csdl.reshape(pivot_point, (1, 3))
#     pivot_rows = csdl.matmat(np.ones((points.shape[0], 1), dtype=float), pivot_point)
#     centered_points = points - pivot_rows

#     x_coordinates = centered_points[csdl.slice[:, 0:1]]
#     y_coordinates = centered_points[csdl.slice[:, 1:2]]
#     z_coordinates = centered_points[csdl.slice[:, 2:3]]

#     angle_radians = angle_degrees * (np.pi / 180.0)
#     cosine = csdl.cos(angle_radians)
#     sine = csdl.sin(angle_radians)

#     rotated_x = cosine * x_coordinates + sine * z_coordinates
#     rotated_z = -sine * x_coordinates + cosine * z_coordinates
#     rotated_points = csdl.concatenate((rotated_x, y_coordinates, rotated_z), axis=1)
#     return rotated_points + pivot_rows


# def _smoothstep_numpy(values: np.ndarray) -> np.ndarray:
#     values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
#     return values * values * (3.0 - 2.0 * values)


# def _apply_stacked_coefficients_csdl(
#     function_set: lfs.FunctionSet,
#     stacked_coefficients: csdl.Variable,
# ) -> None:
#     offset = 0
#     for function in function_set.functions.values():
#         coefficient_shape = function.coefficients.shape
#         num_rows = int(np.prod(coefficient_shape) // 3)
#         function.coefficients = stacked_coefficients[
#             csdl.slice[offset : offset + num_rows, :]
#         ].reshape(coefficient_shape)
#         offset += num_rows

# @dataclass(frozen=True)
# class TransportCGEstimateInputsImperial:
#     fuselage_nose_x_ft: Union[float, csdl.Variable]
#     fuselage_tail_x_ft: Union[float, csdl.Variable]
#     cabin_start_x_ft: Union[float, csdl.Variable]
#     cabin_end_x_ft: Union[float, csdl.Variable]
#     cargo_start_x_ft: Union[float, csdl.Variable]
#     cargo_end_x_ft: Union[float, csdl.Variable]
#     wing_root_le_x_ft: Union[float, csdl.Variable]
#     wing_root_te_x_ft: Union[float, csdl.Variable]
#     wing_yehudi_le_x_ft: Union[float, csdl.Variable]
#     wing_yehudi_te_x_ft: Union[float, csdl.Variable]
#     wing_tip_le_x_ft: Union[float, csdl.Variable]
#     wing_tip_te_x_ft: Union[float, csdl.Variable]
#     horizontal_tail_root_le_x_ft: Union[float, csdl.Variable]
#     horizontal_tail_root_te_x_ft: Union[float, csdl.Variable]
#     horizontal_tail_tip_le_x_ft: Union[float, csdl.Variable]
#     horizontal_tail_tip_te_x_ft: Union[float, csdl.Variable]
#     vertical_tail_root_le_x_ft: Union[float, csdl.Variable]
#     vertical_tail_root_te_x_ft: Union[float, csdl.Variable]
#     vertical_tail_tip_le_x_ft: Union[float, csdl.Variable]
#     vertical_tail_tip_te_x_ft: Union[float, csdl.Variable]
#     nacelle_inlet_x_ft: Union[float, csdl.Variable]
#     nacelle_outlet_x_ft: Union[float, csdl.Variable]
#     main_gear_x_ft: Union[float, csdl.Variable]
#     nose_gear_x_ft: Union[float, csdl.Variable]
#     wing_MAC: Union[float, csdl.Variable]
#     fuselage_height: Union[float, csdl.Variable]
#     fuselage_diameter: Union[float, csdl.Variable]
#     wing_sweep_quarter_chord: Union[float, csdl.Variable]
#     htail_sweep_quarter_chord: Union[float, csdl.Variable]
#     tail_moment_arm: Union[float, csdl.Variable]


# def evaluate_E175_geometry_parameterization(
#     fuse_function_set: lfs.FunctionSet,
#     wing_function_set: lfs.FunctionSet,
#     htail_function_set: lfs.FunctionSet,
#     wing_area_dv: csdl.Variable,
#     wing_AR_dv: csdl.Variable,
#     htail_area_dv: csdl.Variable,
#     htail_AR_dv: csdl.Variable,
#     wing_twist_dvs: csdl.Variable,
#     wing_translation_x: csdl.Variable,
#     htail_root_rotation_degrees: csdl.Variable,
#     fuselage_diameter_scale: csdl.Variable,
#     num_ffd_coefficients_chordwise = 8,
#     num_ffd_sections = 11,
#     do_camber_and_thickness = False,
#     wing_camber_dvs: csdl.Variable = None,
#     wing_thickness_dvs: csdl.Variable = None,
# ) -> TransportCGEstimateInputsImperial:
    
#     wing_geometry = lsdo_geo.Geometry(functions=wing_function_set.functions, space=wing_function_set.space)
#     htail_geometry = lsdo_geo.Geometry(functions=htail_function_set.functions, space=htail_function_set.space)
#     fuse_geometry = lsdo_geo.Geometry(functions=fuse_function_set.functions, space=fuse_function_set.space)

#     # wing_geometry.plot()
#     ####################### WING SECTION KEY POINTS #######################
#     wing_LE_center = wing_geometry.project(np.array([10.882, 0., 0.2]), plot=False)
#     wing_TE_center = wing_geometry.project(np.array([16.326, 0., -0.009]), plot=False)

#     wing_LE_yehudi_right = wing_geometry.project(np.array([13.508, 4.863, 0.684]), plot=False)
#     wing_LE_tip_right = wing_geometry.project(np.array([17.324, 11.911, 1.428]), plot=False)
#     wing_TE_yehudi_right = wing_geometry.project(np.array([16.421, 4.863, 0.623]), plot=False)
#     wing_TE_tip_right = wing_geometry.project(np.array([18.314, 11.911, 1.391]), plot=False)

#     wing_LE_yehudi_left = wing_geometry.project(np.array([13.508, -4.863, 0.684]), plot=False)
#     wing_LE_tip_left = wing_geometry.project(np.array([17.324, -11.911, 1.428]), plot=False)
#     wing_TE_yehudi_left = wing_geometry.project(np.array([16.421, -4.863, 0.623]), plot=False)
#     wing_TE_tip_left = wing_geometry.project(np.array([18.314, -11.911, 1.391]), plot=False)

#     ####################### TAIL SECTION KEY POINTS #######################
#     htail_LE_center = htail_geometry.project(np.array([26., 0., 1.572]), plot=False)
#     htail_TE_center = htail_geometry.project(np.array([29.534, 0., 1.572]), plot=False)

#     htail_LE_tip_right = htail_geometry.project(np.array([29.7, 5.0, 1.572]), plot=False)
#     htail_TE_tip_right = htail_geometry.project(np.array([30.816, 5.0, 1.572]), plot=False)

#     htail_LE_tip_left = htail_geometry.project(np.array([29.7, -5.0, 1.572]), plot=False)
#     htail_TE_tip_left = htail_geometry.project(np.array([30.816, -5.0, 1.572]), plot=False)

#     ####################### FUSELAGE SECTION KEY POINTS #######################
#     fuse_tail_pt = fuse_geometry.project(np.array([31.680, 0., 1.901]), plot=False)
#     fuse_nose_pt = fuse_geometry.project(np.array([0., 0., 0.]), plot=False)

#     cabin_start_pt = fuse_geometry.project(np.array([6., 0., -0.5]), plot=False)
#     cabin_end_pt = fuse_geometry.project(np.array([25.56, 0., 0.03]), plot=False)

#     cargo_start_pt = fuse_geometry.project(np.array([10.0, 0., -0.5]), plot=False)
#     cargo_end_pt = fuse_geometry.project(np.array([20.0, 0., -0.5]), plot=False)

#     nose_landing_gear_pt = fuse_geometry.project(np.array([3.9, 0., -0.5]), plot=False)
#     main_landing_gear_pt = fuse_geometry.project(np.array([16.25, 0., -0.6]), plot=False)

#     fuselage_height_1 = fuse_geometry.project(np.array([18.025, 0., 2.625]), plot=False)
#     fuselage_height_2 = fuse_geometry.project(np.array([18.025, 0., -0.725]), plot=False)

#     fuselage_diameter_1 = fuse_geometry.project(np.array([18.025, -1.505, 0.950]), plot=False)
#     fuselage_diameter_2 = fuse_geometry.project(np.array([18.025, 1.505, 0.950]), plot=False)


#     num_semispan_ffd_sections = num_ffd_sections // 2 + 1
#     # Note: This FFD block construction is one of a few helper functions that can be used to create a FFD block.
#     #       The "manual" method is to use construct_ffd_block_from_corners, which allows for defining the coefficients directly.
#     ffd_block = construct_ffd_block_around_entities(
#         entities=wing_geometry,
#         # num_coefficients=(num_ffd_coefficients_chordwise, num_ffd_sections, 2),
#         num_coefficients=(2, num_ffd_sections, 2),
#         # degree=(3, 3, 1),
#         degree=(1, 1, 1),
#     )

#     num_ffd_sections_htail = 3
#     ffd_block_htail = construct_ffd_block_around_entities(
#         entities=htail_geometry,
#         num_coefficients=(2, num_ffd_sections_htail, 2),
#         degree=(1, 1, 1),
#     )

#     ffd_sectional_parameterization = VolumeSectionalParameterization(
#         name="ffd_sectional_parameterization",
#         parameterized_points=ffd_block.coefficients,
#         principal_parametric_dimension=1,
#     )

#     ffd_sectional_parameterization_htail = VolumeSectionalParameterization(
#         name="ffd_sectional_parameterization_htail",
#         parameterized_points=ffd_block_htail.coefficients,
#         principal_parametric_dimension=1,
#     )

#     space_of_linear_2_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(1,), coefficients_shape=(2,))
#     space_of_cubic_8_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(3,), coefficients_shape=(8,))
#     space_of_linear_3_dof_b_splines = lfs.BSplineSpaceNew(num_parametric_dimensions=1, degree=(1,), coefficients_shape=(3,))

#     # Chord stretch is parameterized on the right semispan and mirrored so the solver
#     # cannot create a left/right asymmetric chord distribution.
#     chord_stretching_b_spline = lfs.Function(space=space_of_linear_2_dof_b_splines,
#                                             coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])), name='chord_stretching_b_spline_coefficients')

#     # Span translation is parameterized with a smooth semispan B-spline and mirrored
#     # antisymmetrically so the root stays on the symmetry plane.
#     wingspan_stretching_b_spline = lfs.Function(
#         # space=space_of_cubic_8_dof_b_splines,
#         space=space_of_linear_3_dof_b_splines,
#         # coefficients=csdl.Variable(shape=(8,), value=np.array([0., 0., 0., 0., 0., 0., 0., 0.])),
#         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
#         name='wingspan_stretching_b_spline_coefficients',
#     )

#     wing_sweep_b_spline = lfs.Function(
#         space=space_of_linear_3_dof_b_splines,
#         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
#         name='wing_sweep_b_spline_coefficients',
#     )

#     tail_span_stretching_b_spline = lfs.Function(
#         space=space_of_linear_3_dof_b_splines,
#         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
#         name='tail_span_stretching_b_spline_coefficients',
#     )

#     tail_chord_stretching_b_spline = lfs.Function(
#         space=space_of_linear_2_dof_b_splines,
#         coefficients=csdl.Variable(shape=(2,), value=np.array([0., 0.])),
#         name='tail_chord_stretching_b_spline_coefficients',
#     )

#     tail_sweep_b_spline = lfs.Function(
#         space=space_of_linear_3_dof_b_splines,
#         coefficients=csdl.Variable(shape=(3,), value=np.array([0., 0., 0.])),
#         name='tail_sweep_b_spline_coefficients',
#     )

#     # The full-wing FFD sections run from left tip -> root -> right tip, so use a semispan
#     # twist profile (root -> tip) and mirror it to get a symmetric nonlinear distribution.
#     twist_b_spline = lfs.Function(
#         space=space_of_linear_2_dof_b_splines,
#         # space=space_of_cubic_8_dof_b_splines,
#         coefficients=wing_twist_dvs, # value=np.linspace(10, -10, 8) * np.pi / 180.0,
#         name='twist_b_spline_coefficients',
#     )


#     semispan_parametric_b_spline_inputs = np.linspace(0.0, 1.0, num_semispan_ffd_sections).reshape((-1, 1))
#     right_semispan_chord_stretch_sectional_parameters = chord_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs)
#     chord_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
#     chord_stretch_sectional_parameters = chord_stretch_sectional_parameters.set(
#         csdl.slice[num_ffd_sections//2:],
#         right_semispan_chord_stretch_sectional_parameters,
#     )
#     chord_stretch_sectional_parameters = chord_stretch_sectional_parameters.set(
#         csdl.slice[:num_ffd_sections//2],
#         right_semispan_chord_stretch_sectional_parameters[:0:-1],
#     )

#     wingspan_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
#     right_semispan_wingspan_stretch_sectional_parameters = wingspan_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs)
#     wingspan_stretch_sectional_parameters = wingspan_stretch_sectional_parameters.set(
#         csdl.slice[num_ffd_sections//2:],
#         right_semispan_wingspan_stretch_sectional_parameters,
#     )
#     wingspan_stretch_sectional_parameters = wingspan_stretch_sectional_parameters.set(
#         csdl.slice[:num_ffd_sections//2],
#         -right_semispan_wingspan_stretch_sectional_parameters[:0:-1],
#     )

#     right_semispan_wing_sweep_translation_sectional_parameters = wing_sweep_b_spline.evaluate(
#         semispan_parametric_b_spline_inputs
#     )
#     wing_sweep_translation_sectional_parameters = csdl.Variable(
#         shape=(num_ffd_sections,),
#         value=0.,
#     )
#     wing_sweep_translation_sectional_parameters = (
#         wing_sweep_translation_sectional_parameters.set(
#             csdl.slice[num_ffd_sections//2:],
#             right_semispan_wing_sweep_translation_sectional_parameters,
#         )
#     )
#     wing_sweep_translation_sectional_parameters = (
#         wing_sweep_translation_sectional_parameters.set(
#             csdl.slice[:num_ffd_sections//2],
#             right_semispan_wing_sweep_translation_sectional_parameters[:0:-1],
#         )
#     )

#     right_semispan_twist_sectional_parameters = twist_b_spline.evaluate(semispan_parametric_b_spline_inputs)
#     wing_twist_sectional_parameters = csdl.Variable(shape=(num_ffd_sections,), value=0.)
#     wing_twist_sectional_parameters = wing_twist_sectional_parameters.set(
#         csdl.slice[num_ffd_sections//2:],
#         right_semispan_twist_sectional_parameters,
#     )
#     wing_twist_sectional_parameters = wing_twist_sectional_parameters.set(
#         csdl.slice[:num_ffd_sections//2],
#         right_semispan_twist_sectional_parameters[:0:-1],
#     )

#     # tail 
#     semispan_parametric_b_spline_inputs_tail = np.linspace(0.0, 1.0, 2).reshape((-1, 1))

#     right_semispan_tail_chord_stretch_sectional_parameters = tail_chord_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
#     chord_stretch_sectional_parameters_tail = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
#     chord_stretch_sectional_parameters_tail = chord_stretch_sectional_parameters_tail.set(
#         csdl.slice[1:],
#         right_semispan_tail_chord_stretch_sectional_parameters,
#     )
#     chord_stretch_sectional_parameters_tail = chord_stretch_sectional_parameters_tail.set(
#         csdl.slice[:1],
#         right_semispan_tail_chord_stretch_sectional_parameters[:0:-1],
#     )

#     right_semispan_tail_span_stretch_sectional_parameters = tail_span_stretching_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
#     htail_span_stretch_sectional_parameters = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
#     htail_span_stretch_sectional_parameters = htail_span_stretch_sectional_parameters.set(
#         csdl.slice[1:],
#         right_semispan_tail_span_stretch_sectional_parameters,
#     )
#     htail_span_stretch_sectional_parameters = htail_span_stretch_sectional_parameters.set(
#         csdl.slice[:1],
#         -right_semispan_tail_span_stretch_sectional_parameters[:0:-1],
#     )

#     right_semispan_tail_sweep_translation_sectional_parameters = tail_sweep_b_spline.evaluate(semispan_parametric_b_spline_inputs_tail)
#     tail_sweep_translation_sectional_parameters = csdl.Variable(shape=(num_ffd_sections_htail,), value=0.)
#     tail_sweep_translation_sectional_parameters = tail_sweep_translation_sectional_parameters.set(
#         csdl.slice[1:],
#         right_semispan_tail_sweep_translation_sectional_parameters,
#     )
#     tail_sweep_translation_sectional_parameters = tail_sweep_translation_sectional_parameters.set(
#         csdl.slice[:1],
#         right_semispan_tail_sweep_translation_sectional_parameters[:0:-1],
#     )

#     # Evaluate the sectional parameterization to get the FFD coefficients
#     sectional_parameters = VolumeSectionalParameterizationInputs()
#     sectional_parameters.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters)
#     sectional_parameters.add_sectional_translation(axis=0, translation=wing_sweep_translation_sectional_parameters)
#     sectional_parameters.add_sectional_translation(axis=1, translation=wingspan_stretch_sectional_parameters)
#     sectional_parameters.add_sectional_rotation(axis=1, rotation=wing_twist_sectional_parameters)

#     sectional_parameters_htail = VolumeSectionalParameterizationInputs()
#     sectional_parameters_htail.add_sectional_stretch(axis=0, stretch=chord_stretch_sectional_parameters_tail)
#     sectional_parameters_htail.add_sectional_translation(axis=1, translation=htail_span_stretch_sectional_parameters)
#     sectional_parameters_htail.add_sectional_translation(axis=0, translation=tail_sweep_translation_sectional_parameters)


#     ffd_coefficients = ffd_sectional_parameterization.evaluate(sectional_parameters, plot=False)
#     ffd_coefficients_htail = ffd_sectional_parameterization_htail.evaluate(sectional_parameters_htail, plot=False)

#     # Parameterize each spanwise FFD section as a rectangular airfoil in the local
#     # thickness direction "w". Interior chordwise rows and non-winglet spanwise
#     # sections get percent camber and percent thickness changes based on the local
#     # section chord length.
    

#     # Example initialization for debugging:
#     # base_delta_camber_percent_design_dof_values = np.array([
#     #     [0.0, 0.1, 0.2, 0.3],
#     #     [0.0, 0.2, 0.3, 0.4],
#     #     [0.0, 0.3, 0.5, 0.6],
#     #     [0.0, 0.3, 0.5, 0.6],
#     #     [0.0, 0.2, 0.3, 0.4],
#     #     [0.0, 0.1, 0.2, 0.3],
#     # ])
#     # delta_camber_percent_design_dof_values = np.vstack([
#     #     np.interp(
#     #         np.linspace(0.0, 1.0, num_semispan_ffd_sections - 1),
#     #         np.linspace(0.0, 1.0, base_delta_camber_percent_design_dof_values.shape[1]),
#     #         base_delta_camber_percent_design_dof_values[row_index],
#     #     )
#     #     for row_index in range(base_delta_camber_percent_design_dof_values.shape[0])
#     # ])
#     # delta_camber_percent_design_dof = csdl.Variable(
#     #     shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
#     #     # value=delta_camber_percent_design_dof_values,
#     #     # value=np.zeros((num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1)),
#     #     value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 10,
#     #     name='delta_camber_percent_design_dof',
#     # )
#     # delta_thickness_percent_design_dof = csdl.Variable(
#     #     shape=(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1),
#     #     # value=np.zeros((num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1)),
#     #     value=np.random.rand(num_ffd_coefficients_chordwise - 2, num_semispan_ffd_sections - 1) * 5,
#     #     name='delta_thickness_percent_design_dof',
#     # )
#     # delta_camber_percent_design_dof.set_as_design_variable(lower=-10.0, upper=10.0, scaler=0.1)
#     # delta_thickness_percent_design_dof.set_as_design_variable(lower=-5.0, upper=5.0, scaler=0.2)

#     if do_camber_and_thickness:
#         delta_camber_percent = csdl.Variable(
#             shape=(num_ffd_coefficients_chordwise, num_ffd_sections),
#             value=0.,
#         )
#         delta_thickness_percent = csdl.Variable(
#             shape=(num_ffd_coefficients_chordwise, num_ffd_sections),
#             value=0.,
#         )

#         delta_camber_percent = delta_camber_percent.set(
#             csdl.slice[1:-1, num_ffd_sections//2:-1],
#             wing_camber_dvs,
#         )
#         delta_camber_percent = delta_camber_percent.set(
#             csdl.slice[1:-1, 1:num_ffd_sections//2],
#             wing_camber_dvs[:, 1:][:, ::-1],
#         )
#         delta_thickness_percent = delta_thickness_percent.set(
#             csdl.slice[1:-1, num_ffd_sections//2:-1],
#             wing_thickness_dvs,
#         )
#         delta_thickness_percent = delta_thickness_percent.set(
#             csdl.slice[1:-1, 1:num_ffd_sections//2],
#             wing_thickness_dvs[:, 1:][:, ::-1],
#         )

#         local_w_vectors = ffd_coefficients[:, :, 1, :] - ffd_coefficients[:, :, 0, :]
#         local_w_gap = csdl.norm(local_w_vectors, axes=(2,))
#         local_w_hat = local_w_vectors / csdl.expand(local_w_gap, local_w_vectors.shape, 'ij->ija')

#         sectional_chord_length = 0.5 * (
#             csdl.norm(
#                 ffd_coefficients[-1, :, 0, :] - ffd_coefficients[0, :, 0, :],
#                 axes=(1,),
#             )
#             + csdl.norm(
#                 ffd_coefficients[-1, :, 1, :] - ffd_coefficients[0, :, 1, :],
#                 axes=(1,),
#             )
#         )
#         sectional_chord_length = csdl.expand(
#             sectional_chord_length,
#             (num_ffd_coefficients_chordwise, num_ffd_sections),
#             'j->ij',
#         )

#         delta_camber = (delta_camber_percent / 100.0) * sectional_chord_length
#         delta_thickness = (delta_thickness_percent / 100.0) * sectional_chord_length

#         top_w_displacement = delta_camber + 0.5 * delta_thickness
#         bottom_w_displacement = delta_camber - 0.5 * delta_thickness

#         top_vector_displacement = csdl.expand(
#             top_w_displacement, local_w_vectors.shape, 'ij->ija'
#         ) * local_w_hat
#         bottom_vector_displacement = csdl.expand(
#             bottom_w_displacement, local_w_vectors.shape, 'ij->ija'
#         ) * local_w_hat

#         ffd_coefficients = ffd_coefficients.set(
#             csdl.slice[:, :, 1, :],
#             ffd_coefficients[:, :, 1, :] + top_vector_displacement,
#         )
#         ffd_coefficients = ffd_coefficients.set(
#             csdl.slice[:, :, 0, :],
#             ffd_coefficients[:, :, 0, :] + bottom_vector_displacement,
#         )

#     # updated_local_w_gap = csdl.norm(
#     #     ffd_coefficients[:, :, 1, :] - ffd_coefficients[:, :, 0, :],
#     #     axes=(2,),
#     # )
#     # minimum_ffd_layer_gap = csdl.minimum(updated_local_w_gap)

#     # Evaluate the FFD and set the coefficients of the geometry
#     wing_geometry_coefficients = ffd_block.evaluate_ffd(coefficients=ffd_coefficients, plot=False)
#     wing_geometry.set_coefficients(wing_geometry_coefficients) 
#     # ffd_block.plot()

#     htail_geometry_coefficients = ffd_block_htail.evaluate_ffd(coefficients=ffd_coefficients_htail, plot=False)
#     htail_geometry.set_coefficients(htail_geometry_coefficients) 
#     # ffd_block_htail.plot()

#     # wing_geometry.plot()
#     # exit()

#     # Wing area reference area computation
#     # One side of the wing area is estimated as the area of two trapezoids: 
#     #  - one with bases defined by the root chord and the yehudi chord, 
#     #  - one with bases defined by the yehudi chord and the tip chord. 
#     # The total area is then twice this area to account for both sides of the wing.

#     # Trapz area for center to yehudi section, right semispan
#     trapz_base_wing_center = csdl.norm(wing_geometry.evaluate(wing_LE_center) - wing_geometry.evaluate(wing_TE_center)) 
#     trapz_base_wing_yehudi = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_right) - wing_geometry.evaluate(wing_TE_yehudi_right)) 
#     # trapz height is the spanwise distance between the center and yehudi sections
#     trapz_height_wing_yehudi = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_right)[1] - wing_geometry.evaluate(wing_LE_center)[1]) 
#     trapz_area_wing_yehudi = 0.5 * (trapz_base_wing_center + trapz_base_wing_yehudi) * trapz_height_wing_yehudi

#     # Trapz area for yehudi to tip section
#     trapz_base_wing_tip = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_TE_tip_right)) 
#     # trapz height is the spanwise distance between the yehudi and tip sections
#     trapz_height_wing_tip = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right)[1] - wing_geometry.evaluate(wing_LE_yehudi_right)[1]) 
#     trapz_area_wing_tip = 0.5 * (trapz_base_wing_yehudi + trapz_base_wing_tip) * trapz_height_wing_tip

#     # Left semispan area is computed explicitly so the area metric remains meaningful
#     # even if some future parameterization accidentally introduces asymmetry.
#     trapz_base_wing_yehudi_left = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_left) - wing_geometry.evaluate(wing_TE_yehudi_left))
#     trapz_height_wing_yehudi_left = csdl.norm(wing_geometry.evaluate(wing_LE_yehudi_left)[1] - wing_geometry.evaluate(wing_LE_center)[1])
#     trapz_area_wing_yehudi_left = 0.5 * (trapz_base_wing_center + trapz_base_wing_yehudi_left) * trapz_height_wing_yehudi_left

#     trapz_base_wing_tip_left = csdl.norm(wing_geometry.evaluate(wing_LE_tip_left) - wing_geometry.evaluate(wing_TE_tip_left))
#     trapz_height_wing_tip_left = csdl.norm(wing_geometry.evaluate(wing_LE_tip_left)[1] - wing_geometry.evaluate(wing_LE_yehudi_left)[1])
#     trapz_area_wing_tip_left = 0.5 * (trapz_base_wing_yehudi_left + trapz_base_wing_tip_left) * trapz_height_wing_tip_left

#     estimated_wing_area = trapz_area_wing_yehudi + trapz_area_wing_tip + trapz_area_wing_yehudi_left + trapz_area_wing_tip_left
#     print("Estimated Wing Area: ", estimated_wing_area.value)

#     wing_span = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_LE_tip_left))
#     wing_AR = wing_span**2 / estimated_wing_area
#     print("Estimated Wing Aspect Ratio: ", wing_AR.value)


#     # fuselage-htail connection constraints
#     fuse_htail_connection_constraint = csdl.norm(fuse_geometry.evaluate(fuse_tail_pt) - htail_geometry.evaluate(htail_TE_center))

#     # htail area reference area computation
#     htail_trapz_base_htail_center = csdl.norm(htail_geometry.evaluate(htail_LE_center) - htail_geometry.evaluate(htail_TE_center))
#     htail_trapz_base_htail_tip_right = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right) - htail_geometry.evaluate(htail_TE_tip_right))
#     htail_trapz_height_htail_tip_right = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right)[1] - htail_geometry.evaluate(htail_LE_center)[1])
#     htail_trapz_area_htail_tip_right = 0.5 * (htail_trapz_base_htail_center + htail_trapz_base_htail_tip_right) * htail_trapz_height_htail_tip_right

#     htail_trapz_base_htail_tip_left = csdl.norm(htail_geometry.evaluate(htail_LE_tip_left) - htail_geometry.evaluate(htail_TE_tip_left))
#     htail_trapz_height_htail_tip_left = csdl.norm(htail_geometry.evaluate(htail_LE_tip_left)[1] - htail_geometry.evaluate(htail_LE_center)[1])
#     htail_trapz_area_htail_tip_left = 0.5 * (htail_trapz_base_htail_center + htail_trapz_base_htail_tip_left) * htail_trapz_height_htail_tip_left

#     htail_area = htail_trapz_area_htail_tip_right + htail_trapz_area_htail_tip_left
#     print("Estimated Htail Area: ", htail_area.value)

#     htail_span = csdl.norm(htail_geometry.evaluate(htail_LE_tip_right) - htail_geometry.evaluate(htail_LE_tip_left))
#     htail_AR = htail_span**2 / htail_area
#     print("Estimated Htail Aspect Ratio: ", htail_AR.value)

#     geometry_solver = ParameterizationSolver()

#     # Define the states for the parameterization solver (solver will manipulate these to
#     # achieve the variables). The chord and span states are already mirrored, so the
#     # inner solve cannot create a left/right asymmetric wing.
#     geometry_solver.add_state(chord_stretching_b_spline.coefficients)
#     geometry_solver.add_state(wingspan_stretching_b_spline.coefficients)
#     geometry_solver.add_state(wing_sweep_b_spline.coefficients)
#     geometry_solver.add_state(tail_span_stretching_b_spline.coefficients)
#     geometry_solver.add_state(tail_chord_stretching_b_spline.coefficients)
#     geometry_solver.add_state(tail_sweep_b_spline.coefficients)

#     # wing_area_dv = csdl.Variable(shape=(1,), name='wing_area', value=np.array([70.]))
#     # wing_area_dv.set_as_design_variable(lower=0.75 * 70., upper=1.25 * 70., scaler=0.02)
#     # wing_AR_dv = csdl.Variable(shape=(1,), name='wing_AR', value=np.array([8.4]))
#     # wing_AR_dv.set_as_design_variable(lower=0.75 * 8.4, upper=1.25 * 8.4, scaler=0.15)

#     # htail_area_dv = csdl.Variable(shape=(1,), name='htail_area', value=np.array([23.25]))
#     # htail_area_dv.set_as_design_variable(lower=0.75 * 23.25, upper=1.25 * 23.25, scaler=0.05)
#     # htail_AR_dv = csdl.Variable(shape=(1,), name='htail_AR', value=np.array([4.3]))
#     # htail_AR_dv.set_as_design_variable(lower=0.75 * 4.3, upper=1.25 * 4.3, scaler=0.3)

#     geometric_variables = GeometricVariables()
#     geometric_variables.add_variable(estimated_wing_area, wing_area_dv, penalty_value=None)
#     geometric_variables.add_variable(wing_AR, wing_AR_dv, penalty_value=None)
#     geometric_variables.add_variable(fuse_htail_connection_constraint, fuse_htail_connection_constraint.value, penalty_value=None)
#     geometric_variables.add_variable(htail_area, htail_area_dv, penalty_value=None)
#     geometric_variables.add_variable(htail_AR, htail_AR_dv, penalty_value=None)

#     geometry_solver.evaluate(geometric_variables)

#     fuselage_diameter_taper_before_empennage = True
#     fuselage_diameter_taper_margin_htail_chords = 0.50
#     fuselage_diameter_taper_length_htail_chords = 2.00

#     # Apply the rigid-body/component-level post-solve motions here so the
#     # downstream movement code consumes the already-parameterized geometry.
#     wing_coefficients_csdl = _stack_coefficients_csdl(wing_function_set)
#     wing_translation_vector = csdl.reshape(
#         csdl.concatenate(
#             (wing_translation_x, np.zeros((1,), dtype=float), np.zeros((1,), dtype=float)),
#             axis=0,
#         ),
#         (1, 3),
#     )
#     wing_translation_rows = csdl.matmat(
#         np.ones((wing_coefficients_csdl.shape[0], 1), dtype=float),
#         wing_translation_vector,
#     )
#     _apply_stacked_coefficients_csdl(
#         wing_function_set,
#         wing_coefficients_csdl + wing_translation_rows,
#     )

#     htail_root_leading_point = htail_geometry.evaluate(htail_LE_center)
#     htail_root_trailing_point = htail_geometry.evaluate(htail_TE_center)
#     htail_root_quarter_chord = htail_root_leading_point + 0.25 * (
#         htail_root_trailing_point - htail_root_leading_point
#     )
#     htail_coefficients_csdl = _stack_coefficients_csdl(htail_function_set)
#     rotated_htail_coefficients = _rotate_points_about_y_axis_csdl(
#         htail_coefficients_csdl,
#         htail_root_rotation_degrees,
#         htail_root_quarter_chord,
#     )
#     _apply_stacked_coefficients_csdl(
#         htail_function_set,
#         rotated_htail_coefficients,
#     )

#     current_fuse_coefficients = _stack_coefficients_numpy(fuse_function_set)
#     current_htail_root_leading_point = np.asarray(
#         htail_root_leading_point.value,
#         dtype=float,
#     ).reshape((1, 3))
#     current_htail_root_trailing_point = np.asarray(
#         htail_root_trailing_point.value,
#         dtype=float,
#     ).reshape((1, 3))
#     htail_root_chord_proxy = max(
#         float(current_htail_root_trailing_point[0, 0] - current_htail_root_leading_point[0, 0]),
#         1e-12,
#     )
#     fuselage_centerline_point = np.array(
#         [
#             0.0,
#             0.5
#             * (
#                 float(np.min(current_fuse_coefficients[:, 1]))
#                 + float(np.max(current_fuse_coefficients[:, 1]))
#             ),
#             0.5
#             * (
#                 float(np.min(current_fuse_coefficients[:, 2]))
#                 + float(np.max(current_fuse_coefficients[:, 2]))
#             ),
#         ],
#         dtype=float,
#     )
#     fuselage_centerline_rows = np.broadcast_to(
#         fuselage_centerline_point,
#         current_fuse_coefficients.shape,
#     ).copy()
#     fuselage_radial_mask = np.broadcast_to(
#         np.array([0.0, 1.0, 1.0], dtype=float),
#         current_fuse_coefficients.shape,
#     ).copy()
#     fuselage_diameter_taper_x_end = min(
#         float(current_htail_root_leading_point[0, 0]),
#         float(current_htail_root_trailing_point[0, 0]),
#     ) - fuselage_diameter_taper_margin_htail_chords * htail_root_chord_proxy
#     fuselage_diameter_taper_length = (
#         fuselage_diameter_taper_length_htail_chords * htail_root_chord_proxy
#     )
#     if fuselage_diameter_taper_before_empennage:
#         fuselage_diameter_taper_parameter = (
#             fuselage_diameter_taper_x_end - current_fuse_coefficients[:, 0]
#         ) / max(fuselage_diameter_taper_length, 1e-12)
#         fuselage_diameter_taper_weights = _smoothstep_numpy(
#             fuselage_diameter_taper_parameter
#         )
#     else:
#         fuselage_diameter_taper_weights = np.ones(
#             current_fuse_coefficients.shape[0],
#             dtype=float,
#         )
#     fuselage_diameter_taper_rows = np.broadcast_to(
#         fuselage_diameter_taper_weights.reshape((-1, 1)),
#         current_fuse_coefficients.shape,
#     ).copy()
#     fuse_coefficients_csdl = _stack_coefficients_csdl(fuse_function_set)
#     fuselage_radial_offsets_csdl = (
#         fuse_coefficients_csdl - fuselage_centerline_rows
#     ) * fuselage_radial_mask * fuselage_diameter_taper_rows
#     moved_fuse_coefficients = fuse_coefficients_csdl + (
#         fuselage_diameter_scale - 1.0
#     ) * fuselage_radial_offsets_csdl
#     _apply_stacked_coefficients_csdl(
#         fuse_function_set,
#         moved_fuse_coefficients,
#     )


#     print("wing_area", estimated_wing_area.value)
#     print("wing_AR", wing_AR.value)
#     print("wing_twist_sectional_parameters [deg]", wing_twist_sectional_parameters.value * 180.0 / np.pi)
#     print("chord_stretch_sectional_parameters", chord_stretch_sectional_parameters.value)
#     print("wing_sweep_translation_sectional_parameters", wing_sweep_translation_sectional_parameters.value)
#     print("wingspan_stretch_sectional_parameters", wingspan_stretch_sectional_parameters.value)
#     # print("minimum_ffd_layer_gap", minimum_ffd_layer_gap.value)

#     print("htail_area", htail_area.value)
#     print("htail_AR", htail_AR.value)


#     # return cg_estimate_inputs

#     wing_root_chord = csdl.norm(wing_geometry.evaluate(wing_LE_center) - wing_geometry.evaluate(wing_TE_center))
#     wing_tip_chord = csdl.norm(wing_geometry.evaluate(wing_LE_tip_right) - wing_geometry.evaluate(wing_TE_tip_right))
#     taper_ratio = wing_tip_chord / wing_root_chord

#     MAC = (2/3) * wing_root_chord * (1 + taper_ratio + taper_ratio**2) / (1 + taper_ratio)
#     print("Estimated Wing MAC: ", MAC.value)

#     fuselage_height = csdl.norm(fuse_geometry.evaluate(fuselage_height_1) - fuse_geometry.evaluate(fuselage_height_2))
#     fuselage_diameter = csdl.norm(fuse_geometry.evaluate(fuselage_diameter_1) - fuse_geometry.evaluate(fuselage_diameter_2))
#     print("Estimated Fuselage Height: ", fuselage_height.value)
#     print("Estimated Fuselage Diameter: ", fuselage_diameter.value)

#     # compute quarter chord sweep for wing and tail using the leading and trailing edge points at the center and yehudi sections
#     wing_LE_center_point = wing_geometry.evaluate(wing_LE_center)
#     wing_TE_center_point = wing_geometry.evaluate(wing_TE_center)
#     wing_LE_yehudi_right_point = wing_geometry.evaluate(wing_LE_yehudi_right)
#     wing_TE_yehudi_right_point = wing_geometry.evaluate(wing_TE_yehudi_right)
#     wing_quarter_chord_center_point = wing_LE_center_point + 0.25 * (wing_TE_center_point - wing_LE_center_point)
#     wing_quarter_chord_yehudi_right_point = wing_LE_yehudi_right_point + 0.25 * (wing_TE_yehudi_right_point - wing_LE_yehudi_right_point)
#     wing_quarter_chord_sweep = csdl.arctan2(
#         wing_quarter_chord_yehudi_right_point[0] - wing_quarter_chord_center_point[0],
#         wing_quarter_chord_yehudi_right_point[1] - wing_quarter_chord_center_point[1],
#     )

#     htail_LE_center_point = htail_geometry.evaluate(htail_LE_center)
#     htail_TE_center_point = htail_geometry.evaluate(htail_TE_center)
#     htail_LE_yehudi_right_point = htail_geometry.evaluate(htail_LE_tip_right)
#     htail_TE_yehudi_right_point = htail_geometry.evaluate(htail_TE_tip_right)
#     htail_quarter_chord_center_point = htail_LE_center_point + 0.25 * (htail_TE_center_point - htail_LE_center_point)
#     htail_quarter_chord_yehudi_right_point = htail_LE_yehudi_right_point + 0.25 * (htail_TE_yehudi_right_point - htail_LE_yehudi_right_point)
#     htail_quarter_chord_sweep = csdl.arctan2(
#         htail_quarter_chord_yehudi_right_point[0] - htail_quarter_chord_center_point[0],
#         htail_quarter_chord_yehudi_right_point[1] - htail_quarter_chord_center_point[1],
#     )

#     print("Estimated Wing Quarter Chord Sweep: ", wing_quarter_chord_sweep.value * 180.0 / np.pi)
#     print("Estimated Htail Quarter Chord Sweep: ", htail_quarter_chord_sweep.value * 180.0 / np.pi)

#     # tail moment arm
#     tail_moment_arm = csdl.norm(wing_quarter_chord_center_point - htail_quarter_chord_center_point)

#     cg_estimate_inputs = TransportCGEstimateInputsImperial(
#         fuselage_nose_x_ft=fuse_geometry.evaluate(fuse_nose_pt)[0] * M_TO_FT,
#         fuselage_tail_x_ft=fuse_geometry.evaluate(fuse_tail_pt)[0] * M_TO_FT,
#         cabin_start_x_ft=fuse_geometry.evaluate(cabin_start_pt)[0] * M_TO_FT,
#         cabin_end_x_ft=fuse_geometry.evaluate(cabin_end_pt)[0] * M_TO_FT,
#         cargo_start_x_ft=fuse_geometry.evaluate(cargo_start_pt)[0] * M_TO_FT,
#         cargo_end_x_ft=fuse_geometry.evaluate(cargo_end_pt)[0] * M_TO_FT,
#         wing_root_le_x_ft=wing_geometry.evaluate(wing_LE_center)[0] * M_TO_FT,
#         wing_root_te_x_ft=wing_geometry.evaluate(wing_TE_center)[0] * M_TO_FT,
#         wing_yehudi_le_x_ft=wing_geometry.evaluate(wing_LE_yehudi_right)[0] * M_TO_FT,
#         wing_yehudi_te_x_ft=wing_geometry.evaluate(wing_TE_yehudi_right)[0] * M_TO_FT,
#         wing_tip_le_x_ft=wing_geometry.evaluate(wing_LE_tip_right)[0] * M_TO_FT,
#         wing_tip_te_x_ft=wing_geometry.evaluate(wing_TE_tip_right)[0] * M_TO_FT,
#         horizontal_tail_root_le_x_ft=htail_geometry.evaluate(htail_LE_center)[0] * M_TO_FT,
#         horizontal_tail_root_te_x_ft=htail_geometry.evaluate(htail_TE_center)[0] * M_TO_FT,
#         horizontal_tail_tip_le_x_ft=htail_geometry.evaluate(htail_LE_tip_right)[0] * M_TO_FT,
#         horizontal_tail_tip_te_x_ft=htail_geometry.evaluate(htail_TE_tip_right)[0] * M_TO_FT,
#         vertical_tail_root_le_x_ft=26.1 * M_TO_FT,
#         vertical_tail_root_te_x_ft=31.2 * M_TO_FT,
#         vertical_tail_tip_le_x_ft=29.3 * M_TO_FT,
#         vertical_tail_tip_te_x_ft=31.0 * M_TO_FT,
#         nacelle_inlet_x_ft=(wing_geometry.evaluate(wing_LE_yehudi_right)[0] - 2.9) * M_TO_FT,
#         nacelle_outlet_x_ft=(wing_geometry.evaluate(wing_TE_yehudi_right)[0] - 3.1) * M_TO_FT,
#         main_gear_x_ft=fuse_geometry.evaluate(main_landing_gear_pt)[0] * M_TO_FT,
#         nose_gear_x_ft=fuse_geometry.evaluate(nose_landing_gear_pt)[0] * M_TO_FT,
#         wing_MAC=MAC,
#         wing_sweep_quarter_chord=wing_quarter_chord_sweep,
#         htail_sweep_quarter_chord=htail_quarter_chord_sweep,
#         fuselage_diameter=fuselage_diameter * M_TO_FT,
#         fuselage_height=fuselage_height * M_TO_FT,
#         tail_moment_arm=tail_moment_arm,
#     )

#     # print("fuselage_diameter_ft", cg_estimate_inputs.fuselage_diameter.value)
#     # print("fuselage_height_ft", cg_estimate_inputs.fuselage_height.value)
#     # exit("hi")

#     return cg_estimate_inputs
    

#     ######################### END EMBRAER 175 EXAMPLE CODE #########################
