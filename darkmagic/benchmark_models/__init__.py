from darkmagic.benchmark_models.anapole import anapole as anapole
from darkmagic.benchmark_models.hadrophilic_scalar_mediator import (
    heavy_scalar_mediator as heavy_scalar_mediator,
)
from darkmagic.benchmark_models.hadrophilic_scalar_mediator import (
    light_scalar_mediator as light_scalar_mediator,
)
from darkmagic.benchmark_models.magnetic_dipole import (
    magnetic_dipole as magnetic_dipole,
)

BUILT_IN_MODELS = {
    model.shortname: model
    for model in [
        light_scalar_mediator,
        heavy_scalar_mediator,
        magnetic_dipole,
        anapole,
    ]
}
