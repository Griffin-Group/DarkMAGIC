import pytest
from pytest_parametrize_cases import Case, parametrize_cases

from darkmagic import Model
from darkmagic.model import (
    SUPPORTED_OPERATORS,
    ExtraCoefficientFunctionWarning,
    MissingCoefficientFunctionException,
    UnsupportedOperatorException,
)


def one_func(q, m_chi, spin):
    return 1.0


def get_model(
    particles,
    prefactors,
    operators,
    zero_coeff_gets_function=False,
    every_operator_has_function=False,
    operators_to_remove_funcs=None,
    particles_to_remove_funcs=None,
):
    """
    Generates a model with the given particles, having certain coefficient prefactors,
    and only including certain operators. if zero_coeff_gets_function = True, particles with a zero prefactor are assigned a coefficient function (for testing warnings).
    """
    coeff_prefactor = {op: {p: prefactors[p] for p in particles} for op in operators}
    if every_operator_has_function:
        operators = SUPPORTED_OPERATORS
    if zero_coeff_gets_function:
        coeff_func = {op: {p: one_func for p in particles} for op in operators}
    else:
        coeff_func = {
            op: {p: one_func for p in particles if prefactors[p] != 0}
            for op in operators
        }
    if particles_to_remove_funcs:
        for p in particles_to_remove_funcs:
            for value in coeff_func.values():
                value.pop(p, None)
    if operators_to_remove_funcs:
        for op in operators_to_remove_funcs:
            coeff_func.pop(op, None)

    return Model("test", coeff_prefactor, coeff_func)


@parametrize_cases(
    Case(
        "everything_nonzero",
        particles={"e", "p", "n"},
        prefactors={"e": 1, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS,
        expected_particles={"e", "p", "n"},
        expected_operators=SUPPORTED_OPERATORS,
    ),
    Case(
        "electron_zero",
        particles={"e", "p", "n"},
        prefactors={"e": 0, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS - {"1", "2", "3"},
        expected_particles={"p", "n"},
        expected_operators=SUPPORTED_OPERATORS - {"1", "2", "3"},
    ),
)
def test_good_models(
    particles, prefactors, operators, expected_particles, expected_operators
):
    model = get_model(
        particles,
        prefactors,
        operators,
    )
    assert model.particles == expected_particles
    assert model.operators == expected_operators


@parametrize_cases(
    Case(
        "warn_extra_particle_func",
        particles={"e", "p", "n"},
        prefactors={"e": 0, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS,
        expected_particles={"p", "n"},
        expected_operators=SUPPORTED_OPERATORS,
    ),
    Case(
        "warn_extra_operator_func",
        particles={"e", "p", "n"},
        prefactors={"e": 1, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS - {"1", "2", "15a"},
        expected_particles={"e", "p", "n"},
        expected_operators=SUPPORTED_OPERATORS - {"1", "2", "15a"},
    ),
)
def test_warning_models(
    particles, prefactors, operators, expected_particles, expected_operators
):
    with pytest.warns(ExtraCoefficientFunctionWarning):
        model = get_model(
            particles,
            prefactors,
            operators,
            zero_coeff_gets_function=True,
            every_operator_has_function=True,
        )
    assert model.particles == expected_particles
    assert model.operators == expected_operators


@parametrize_cases(
    Case(
        "error_missing_particle_func",
        particles={"e", "p", "n"},
        prefactors={"e": 1, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS,
        operators_to_remove_funcs=None,
        particles_to_remove_funcs={"e"},
    ),
    Case(
        "error_missing_operator_func",
        particles={"e", "p", "n"},
        prefactors={"e": 1, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS,
        operators_to_remove_funcs={"1", "2", "3"},
        particles_to_remove_funcs=None,
    ),
)
def test_missing_coeff_models(
    particles,
    prefactors,
    operators,
    particles_to_remove_funcs,
    operators_to_remove_funcs,
):
    with pytest.raises(MissingCoefficientFunctionException):
        get_model(
            particles,
            prefactors,
            operators,
            operators_to_remove_funcs=operators_to_remove_funcs,
            particles_to_remove_funcs=particles_to_remove_funcs,
        )


@parametrize_cases(
    Case(
        "unsupported_operator",
        particles={"e", "p", "n"},
        prefactors={"e": 1, "p": 1, "n": 1},
        operators=SUPPORTED_OPERATORS | {"unsupported"},
    ),
)
def test_unsupported_operator_models(particles, prefactors, operators):
    with pytest.raises(UnsupportedOperatorException):
        get_model(
            particles,
            prefactors,
            operators,
        )
