"""Setuptools configuration for GAMMA.

GAMMA is normally installed into externally managed multidisciplinary-analysis
environments. In particular, DAFoam supplies a tightly coupled MPI/PETSc/Python
stack that pip must not replace. Consequently this package deliberately has no
automatic ``install_requires`` dependencies. See ``requirements.txt`` and
``HPC_DAFOAM_INSTALL.md`` for the prerequisite audit and safe install command.
"""

from pathlib import Path

from setuptools import find_packages, setup


REPOSITORY_ROOT = Path(__file__).resolve().parent


def get_version() -> str:
    for line in (REPOSITORY_ROOT / "bsm3" / "__init__.py").read_text().splitlines():
        if line.startswith("__version__"):
            delimiter = '"' if '"' in line else "'"
            return line.split(delimiter)[1]
    raise RuntimeError("Unable to find the GAMMA version string.")


setup(
    name="gamma-mdo",
    version=get_version(),
    author="Marius Ruh",
    license="LGPLv3+",
    url="https://github.com/MariusLRuh/GAMMA-MDO",
    description="Geometry-aware mesh movement analysis",
    long_description=(REPOSITORY_ROOT / "README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "bsm3.core.boundary_surface_movement": [
            "embraer_175_no_winglets.stp",
            "cessna208_no_elevator_3.stp",
            "cessna208_3_recombine_new.msh",
            "openvsp_euler_volume_mesh/e175_euler_volume.msh",
            "openvsp_euler_volume_mesh/e175_openvsp_aircraft_wall.msh",
            "openvsp_euler_volume_mesh/e175_openvsp_aircraft_wall.volume_map.npz",
        ],
    },
    python_requires=">=3.9",
    platforms=["any"],
    # Deliberately empty: do not let installing GAMMA modify a DAFoam stack.
    install_requires=[],
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
