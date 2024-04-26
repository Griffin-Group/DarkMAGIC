import pytest
import numpy as np
from darkmagic.material import MaterialProperties
import darkmagic.constants as const
from pytest_parametrize_cases import Case, parametrize_cases


def create_test_dict(n_atoms, rank, zeros=False):
    psi = ["n", "p", "e"]
    f = 0 if zeros else 1
    if rank == 0:
        return {p: f * np.ones(n_atoms) for p in psi}
    elif rank == 1:
        return {p: f * np.ones((n_atoms, 3)) for p in psi}
    elif rank == 2:
        return {p: f * np.ones((n_atoms, 3, 3)) for p in psi}
    else:
        raise ValueError(f"Rank {rank} not supported")


@parametrize_cases(
    Case(
        "AllNone",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    )
)
def test_material_properties_initialization(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialProperties(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
    test_scalar = create_test_dict(n_atoms, 0, zeros=True)
    test_vec = create_test_dict(n_atoms, 1, zeros=True)
    test_tens = create_test_dict(n_atoms, 2, zeros=True)

    m_psi = {
        "e": const.m_e,
        "p": const.m_p,
        "n": const.m_n,
    }

    # Check that it fails validation
    with pytest.raises(AssertionError):
        props.validate_for_phonons(n_atoms)
    with pytest.raises(AssertionError):
        props.validate_for_phonons(n_atoms)

    props._validate_input(n_atoms)

    assert set(props.N.keys()) == set(test_scalar.keys()) and all(
        (props.N[key] == test_scalar[key]).all() for key in props.N.keys()
    )
    assert set(props.S.keys()) == set(test_vec.keys()) and all(
        (props.S[key] == test_vec[key]).all() for key in props.S.keys()
    )
    assert set(props.L.keys()) == set(test_vec.keys()) and all(
        (props.L[key] == test_vec[key]).all() for key in props.L.keys()
    )
    assert set(props.L_dot_S.keys()) == set(test_scalar.keys()) and all(
        (props.L_dot_S[key] == test_scalar[key]).all() for key in props.L_dot_S.keys()
    )
    assert set(props.L_tens_S.keys()) == set(test_tens.keys()) and all(
        (props.L_tens_S[key] == test_tens[key]).all() for key in props.L_tens_S.keys()
    )
    assert (props.lambda_S == np.zeros(n_atoms)).all()
    assert (props.lambda_L == np.zeros(n_atoms)).all()
    assert props.m_psi == m_psi


@parametrize_cases(
    Case(
        "N_only",
        N=create_test_dict(5, 0),
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
    Case(
        "S_only",
        N=None,
        S=create_test_dict(5, 1),
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
    Case(
        "L_only",
        N=None,
        S=None,
        L=create_test_dict(5, 1),
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
    Case(
        "L_dot_S_only",
        N=None,
        S=None,
        L=None,
        L_dot_S=create_test_dict(5, 0),
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
    Case(
        "L_tens_S_only",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=create_test_dict(5, 2),
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
)
def test_material_properties_phonon(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialProperties(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
    props.validate_for_phonons(n_atoms)


# Test magnons
@parametrize_cases(
    Case(
        "lambda_S_only",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=np.ones(5),
        lambda_L=None,
        m_psi=None,
        n_atoms=5,
    ),
    Case(
        "lambda_L_only",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=np.ones(5),
        m_psi=None,
        n_atoms=5,
    ),
)
def test_material_properties_magnon(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialProperties(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
    props.validate_for_magnons(n_atoms)


@parametrize_cases(
    Case(
        "N_wrong_size",
        N=create_test_dict(5, 0),
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "S_wrong_size",
        N=None,
        S=create_test_dict(5, 1),
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "L_wrong_size",
        N=None,
        S=None,
        L=create_test_dict(5, 1),
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "L_dot_S_wrong_size",
        N=None,
        S=None,
        L=None,
        L_dot_S=create_test_dict(5, 0),
        L_tens_S=None,
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "L_tens_S_wrong_size",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=create_test_dict(5, 2),
        lambda_S=None,
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "lambda_S_wrong_size",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=np.ones(5),
        lambda_L=None,
        m_psi=None,
        n_atoms=7,
    ),
    Case(
        "lambda_L_wrong_size",
        N=None,
        S=None,
        L=None,
        L_dot_S=None,
        L_tens_S=None,
        lambda_S=None,
        lambda_L=np.ones(5),
        m_psi=None,
        n_atoms=7,
    ),
)
def test_material_properties_invalid(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialProperties(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
    with pytest.raises(AssertionError):
        props._validate_input(n_atoms)
