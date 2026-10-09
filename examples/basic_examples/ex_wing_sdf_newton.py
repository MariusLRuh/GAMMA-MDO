import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import gamma_mdo
from pathlib import Path
import numpy as np
from typing import Optional

DEFAULT_STEP_PATH = Path(__file__).with_name("swept_wing.stp")


def _plot_slices(
    X: np.ndarray,
    Z: np.ndarray,
    y_stations: np.ndarray,
    field_slices: np.ndarray,
    *,
    output: Optional[Path],
    contour_levels: int=20,
    dpi: int=300,
) -> None:
    # if output is not None:
    #     import matplotlib

    #     matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    num_stations = len(y_stations)
    fig, axes = plt.subplots(
        1,
        num_stations,
        figsize=(5.0 * num_stations, 4.5),
        squeeze=False,
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = axes.ravel()

    data_min = float(np.min(field_slices))
    data_max = float(np.max(field_slices))
    if np.isclose(data_min, data_max):
        data_min -= 0.5
        data_max += 0.5

    if data_min < 0.0 < data_max:
        neg_level_count = max(2, contour_levels // 2 + 1)
        pos_level_count = max(2, contour_levels - neg_level_count + 1)
        neg_levels = np.linspace(data_min, 0.0, neg_level_count)
        pos_levels = np.linspace(0.0, data_max, pos_level_count)
        contour_values = np.concatenate([neg_levels, pos_levels[1:]])
        norm = TwoSlopeNorm(vmin=data_min, vcenter=0.0, vmax=data_max)
    else:
        contour_values = np.linspace(data_min, data_max, contour_levels)
        norm = Normalize(vmin=data_min, vmax=data_max)

    filled = None
    for station_index, axis in enumerate(axes):
        field = field_slices[station_index]
        filled = axis.contourf(
            X,
            Z,
            field,
            levels=contour_values,
            cmap="coolwarm",
            norm=norm,
        )
        axis.contour(X, Z, field, levels=[0.0], colors="k", linewidths=1.5)

        axis.set_title(f"y = {float(y_stations[station_index]):.4f}")
        axis.set_xlabel("x")
        axis.set_aspect("equal", adjustable="box")

    axes[0].set_ylabel("z")
    cbar = fig.colorbar(filled, ax=axes.tolist(), shrink=0.9)
    cbar.set_label(f"Signed distance")
    fig.suptitle("Projection-Based SDF Slices", fontsize=14)

    if output is not None:
        plt.show()
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
        print(f"Saved figure to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    rec = csdl.Recorder(inline=True)
    rec.start()

    check_derivatives = True
    plot_sdf_slices = False

    wing_fun_set = lfs.import_file_patched(file_name=str(DEFAULT_STEP_PATH), parallelize=False)
    
    # Keep derivative-check points off repeated internal knot lines, where
    # surface-normal second derivatives are not unique.
    # u_coords = np.array([0.125, 0.25, 0.5, 0.75, 0.875], dtype=float)
    # v_coords = np.array([0.015625, 0.171875, 0.328125, 0.671875, 0.890625], dtype=float)
    u_coords = np.linspace(0.01, 0.99, 5, dtype=float)
    v_coords = np.linspace(0.01, 0.99, 5, dtype=float)
    U, V = np.meshgrid(u_coords, v_coords, indexing="xy")
    parametric_coordinates = [(0, np.array([u, v])) for u, v in zip(U.ravel(), V.ravel())]

    reshaped_coeffs = []
    for fun_ind, fun in wing_fun_set.functions.items():
        coeffs = fun.coefficients
        reshaped_coeffs.append(csdl.reshape(coeffs, (-1, 3)))

    reshaped_stacked_coeffs = csdl.vstack(reshaped_coeffs)

    # create random points in bounding box of wing
    num_points = 100
    reshape_coeffs_np = reshaped_stacked_coeffs.value
    x_min = np.min(reshape_coeffs_np[:, 0])
    x_max = np.max(reshape_coeffs_np[:, 0])
    y_min = np.min(reshape_coeffs_np[:, 1])
    y_max = np.max(reshape_coeffs_np[:, 1])
    z_min = np.min(reshape_coeffs_np[:, 2])
    z_max = np.max(reshape_coeffs_np[:, 2])

    if check_derivatives:
        # Pick a deterministic subset of coefficients as design variables.
        num_coeffs = reshaped_stacked_coeffs.shape[0]
        num_design_vars = 20
        design_var_indices = np.arange(num_design_vars) + 40
        # np.random.choice(num_coeffs, size=num_design_vars, replace=False)

        for i in range(num_design_vars):
            index = design_var_indices[i]
            coeff_csdl = csdl.Variable(name=f"coeff_{i}", shape=(3, ), value=reshape_coeffs_np[index, :])
            coeff_csdl.set_as_design_variable()
            reshaped_stacked_coeffs = reshaped_stacked_coeffs.set(slices=csdl.slice[index, :], value=coeff_csdl)

        # points = np.random.rand(num_points, 3)
        # points[:, 0] = x_min + points[:, 0] * (x_max - x_min)
        # points[:, 1] = y_min + points[:, 1] * (y_max - y_min)
        # points[:, 2] = z_min + points[:, 2] * (z_max - z_min)
        points = wing_fun_set.evaluate(parametric_coordinates=parametric_coordinates, plot=False).value

        points_csdl = csdl.Variable(name="points", value=points)
        points_csdl.set_as_design_variable()

    # 'project' and 'compute_vjp' of instance below are called in the custom op
    projection_model = gamma_mdo.FunctionSetProjectionModel(
        function_set=wing_fun_set,
        warm_start_nu=25, # controls per-patch triangulation resolution
        warm_start_nv=25,
        sdf=True,
        sdf_sign_mode="normal",
        debug=True,
    )

    if check_derivatives:
        sdf_op = gamma_mdo.FunctionSetClosestDistanceOperation(model=projection_model)
        sdf_values = sdf_op.evaluate(coefficients=reshaped_stacked_coeffs, points=points_csdl)

        first_der = csdl.derivative(sdf_values, points_csdl)

        objective = csdl.sum(first_der)
        objective.set_as_objective()

        jax_sim = csdl.experimental.JaxSimulator(recorder=rec, gpu=False)
        jax_sim.check_optimization_derivatives(step_size=1e-5, raise_on_error=False)
    
    if plot_sdf_slices:
        # create 3 slices of points in the x-z plane (structured grid) at different y values
        nx = 200
        nz = 200
        num_slices = 10
        y_values = np.linspace(y_min, y_max, num_slices)
        points_list = []
        # x_min = 5.2
        # x_max = 6.2
        # z_min = -0.1
        # z_max = 0.1
        for y in y_values:
            x_coords = np.linspace(x_min-1., x_max+1., nx, dtype=float)
            z_coords = np.linspace(z_min-0.5, z_max+0.5, nz, dtype=float)
            X, Z = np.meshgrid(x_coords, z_coords, indexing="xy")
            Y = np.full_like(X, y)
            points_list.append(np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1))
        slice_points = np.vstack(points_list)
        slice_points_csdl = csdl.Variable(name="slice_points", value=slice_points)

        sdf_op = gamma_mdo.FunctionSetClosestDistanceOperation(model=projection_model)
        sdf_values = sdf_op.evaluate(coefficients=reshaped_stacked_coeffs, points=slice_points_csdl)
        
        slice_values = []
        for i in range(num_slices):
            slice_values.append(sdf_values.value[i*nx*nz:(i+1)*nx*nz])
        
        field_slices = np.asarray(slice_values, dtype=float).reshape(num_slices, nz, nx)
        output_path = Path(__file__).with_name("wing_sdf_slices.png")
        _plot_slices(
            X=X,
            Z=Z,
            y_stations=y_values,
            field_slices=field_slices,
            output=output_path,
            contour_levels=100,
            dpi=300,
        )


    # print(sdf_values.value)
    
    # print(reshaped_coeffs.shape)
