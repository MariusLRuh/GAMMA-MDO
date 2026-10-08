from pathlib import Path

import csdl_alpha as csdl
import lsdo_function_spaces as lfs
import meshio

from gamma_mdo.core.projections.function_set_projection_numpy import (
    FunctionSetProjector,
)

DEFAULT_STEP_PATH = Path(__file__).with_name("test_wing_geom.stp")
DEFAULT_MESH_PATH = Path(__file__).with_name("test_wing_mesh.msh")

recorder = csdl.Recorder(inline=True)
recorder.start()

wing_geom = lfs.import_file_patched(file_name=str(DEFAULT_STEP_PATH), parallelize=False)
wing_mesh = meshio.read(str(DEFAULT_MESH_PATH))

mesh_vertices = wing_mesh.points

projector = FunctionSetProjector(
    function_set=wing_geom,
    warm_start_nu=25,  # controls per-patch triangulation resolution
    warm_start_nv=25,
    debug=True,
)

projected_parametric_coordinates = projector.project(mesh_vertices)

wing_geom.evaluate(
    parametric_coordinates=projected_parametric_coordinates,
    plot=True,
)
