"""Differentiable boundary-surface and volume mesh movement.

The package exposes four submodules unconditionally: :mod:`gamma_mdo.component_parameters`
for geometry parameter containers, :mod:`gamma_mdo.core` for the mesh-movement
algorithms, :mod:`gamma_mdo.plotting` for optional visualization helpers, and
:mod:`gamma_mdo.preprocessing` for mesh import, conversion, and export.

A small set of projection helpers is re-exported at package level *only when its
optional research dependencies import successfully*. When they do not, the
import is swallowed and the names are simply absent, so the base package stays
usable. Do not assume ``gamma_mdo.FunctionSetProjectionModel`` and its siblings
exist; guard access with :func:`hasattr`, or import from the module that
defines the name, if your code requires them.

Note that :mod:`gamma_mdo.core.projections` is a regular subpackage whose
``__init__`` re-exports nothing, so importing these names *from* it does not
work. The defining modules
are :mod:`gamma_mdo.core.projections.function_set_closest_distance_custom_op` for
``FunctionSetClosestDistanceOperation`` and ``FunctionSetProjectionModel``,
:mod:`gamma_mdo.core.projections.function_set_evaluation_custom_op` for
``FunctionSetEvaluationModel`` and ``FunctionSetEvaluationOperation``, and
:mod:`gamma_mdo.core.projections.function_set_projection_custom_op` for
``FunctionSetProjectionOperation``. Importing them directly raises whatever the
optional dependency raised, rather than silently omitting the name.
"""

__version__ = "0.3.0a1"

from . import component_parameters, core, plotting, preprocessing

# Projection helpers depend on optional research packages. Keep the base package
# importable when those packages are absent or incompatible.
try:
    from .core.projections.function_set_closest_distance_custom_op import (
        FunctionSetClosestDistanceOperation,
        FunctionSetProjectionModel,
    )
    from .core.projections.function_set_evaluation_custom_op import (
        FunctionSetEvaluationModel,
        FunctionSetEvaluationOperation,
    )
    from .core.projections.function_set_projection_custom_op import (
        FunctionSetProjectionOperation,
    )
except Exception:
    pass
