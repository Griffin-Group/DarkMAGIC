import os

import numpy as np
import pytest
from pytest_parametrize_cases import Case, parametrize_cases

import darkmagic.constants as const
from darkmagic.material import MaterialParameters, PhononMaterial
from darkmagic.numerics import MonkhorstPackGrid

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


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
def test_material_parameters_initialization(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialParameters(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
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
def test_material_parameters_phonon(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialParameters(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
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
def test_material_parameters_magnon(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialParameters(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
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
def test_material_parameters_invalid(
    N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi, n_atoms
):
    props = MaterialParameters(N, S, L, L_dot_S, L_tens_S, lambda_S, lambda_L, m_psi)
    with pytest.raises(AssertionError):
        props._validate_input(n_atoms)


@parametrize_cases(
    Case(
        "hcp_He",
        phonopy_yaml_path=f"{TEST_DIR}/data/hcp_He_1GPa.phonopy.yaml",
        n_atoms=2,
        alat=4.852625323567155 * const.bohr_to_Ang,
        m_atom1=4.00260325415,
        label1="He",
        eps=np.diag([1.065579, 1.065579, 1.065579]),
        bec1=np.zeros((3, 3)),
        opt_freq=[2.5810436703, 2.5810436703, 6.4343430269],
        dE_max=6.4343430269 * 1.5,  # 1.5 * max optical phonon
        q_cut=10 * np.sqrt(4.00260325415 * 6.4343430269),  # sqrt(max mass * dE_max)
    )
)
def test_phonon_material(
    phonopy_yaml_path,
    n_atoms,
    alat,
    m_atom1,
    label1,
    eps,
    bec1,
    opt_freq,
    dE_max,
    q_cut,
):
    props = MaterialParameters(N=create_test_dict(2, 0))
    material = PhononMaterial("test", props, phonopy_yaml_path)
    assert material.name == "test"
    assert material.n_atoms == n_atoms
    assert material.structure.lattice.a == pytest.approx(alat * const.Ang_to_inveV)
    assert material.structure.sites[0].label == label1
    assert material.m_atoms[0] == pytest.approx(m_atom1 * const.amu_to_eV)
    assert np.all(material.epsilon == eps)
    assert np.all(material.born[0] == bec1)
    grid = MonkhorstPackGrid([1, 1, 1], material, shift=False)
    freq, eigvec = material.get_eig(grid, with_eigenvectors=True)
    assert freq.shape == (1, n_atoms * 3)
    assert eigvec.shape == (1, n_atoms * 3, n_atoms, 3)
    # Acoustic modes
    assert np.all(np.abs(freq[0, :2]) < 1e-5)
    # Compare against optical frequencies at gamma point
    opt_freq = np.array(opt_freq)
    assert np.all(np.abs(freq[0, 3:] - opt_freq * const.THz_to_eV) < 1e-5)
    # Eigvecs should have norm 1/sqrt(2)
    assert np.all(np.abs(np.linalg.norm(eigvec, axis=-1) - np.sqrt(2) / 2) < 1e-5)
    assert pytest.approx(material.max_dE) == dE_max * const.THz_to_eV
    assert pytest.approx(material.q_cut) == q_cut * np.sqrt(
        const.amu_to_eV * const.THz_to_eV
    )

    # assert np.allclose(eigvec[0, 0, 0], eigvec11)
