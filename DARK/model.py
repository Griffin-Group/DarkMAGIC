"""
    
    Non-relativistic operators

    Notation also descibed in eft_numeric_formulation_notes

    V_j = V^0_j + v_i V^1_{j i}

    V^0_j = V^00_j + S_k V^01_{j k}
    V^1_{j i} = V^10_{j, i} + S_k V^11_{j i k}

    expansion id's (exp_id) = "00", "01", "10", "11"

"""

import itertools

import numpy as np

import DARK.constants as const
from DARK.constants import levi_civita


# TODO: c_dict and c_dict_form should prob just be merged?
class Model:
    def __init__(
        self,
        name: str,
        c_dict: dict,
        c_dict_form: dict,
        Fmed_power: int = 0,
        power_V: int = 0,
        S_chi: float = 0.5,
    ):
        """
        name: string
        Fmed_power: float, negative power of q in the Fmed term
        power_V: float, power of q in the V term (for special mesh)
        s_chi float, S_chi of DM particle
        """
        self.name = name

        self.Fmed_power = Fmed_power
        self.power_V = power_V
        self.s_chi = S_chi
        self.c_dict = c_dict
        self.c_dict_form = c_dict_form


class Potential:
    def __init__(self):
        self.full_V = {}

    def pot(self, f, a, b):
        # If full_V[a] doesn't exist, make it a dictionary
        if a not in self.full_V:
            self.full_V[a] = {}
        self.full_V[a][b] = f

    def get_V(
        self,
        q,
        op_id,
        exp_id,
        psi,
        material,
        m_chi,
        S_chi,
    ):
        try:
            V = self.full_V[op_id]
        except KeyError:
            print(f"Potential V_{op_id} not found")
        return V.get(exp_id, self.get_zeros(exp_id))(q, psi, material, m_chi, S_chi)

    def get_zeros(self, exp_id):
        def zeros_00(*args):
            return np.zeros(args[2])

        def zeros_01(*args):
            return np.zeros((args[2], 3))

        def zeros_11(*args):
            return np.zeros((args[2], 3, 3))

        if exp_id == "00":
            return zeros_00
        elif exp_id == "01" or exp_id == "10":
            return zeros_01
        elif exp_id == "11":
            return zeros_11

    @pot("1", "00")
    def V1_00(q, psi, material, m_chi, S_chi):

        return material.properties.N[psi]

    @pot("3b", "00")
    def V3b_00(q, psi, material, m_chi, S_chi):

        V = np.zeros(material.n_atoms, dtype=complex)

        q_dir = q / np.linalg.norm(q)

        C = -0.5 * np.linalg.norm(q) ** 2 / material.properties.m_psi[psi] ** 2

        for j in range(material.n_atoms):

            LxS_V = material.properties.L_tens_S[psi][j]

            V[j] = C * (np.trace(LxS_V) - np.dot(q_dir, np.matmul(LxS_V, q_dir)))

        return V

    @pot("3a", "10")
    def V3a_10(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return np.array([C * np.cross(Sj, q) for Sj in material.properties.S[psi]])

    @pot("4", "01")
    def V4_01(q, psi, material, m_chi, S_chi):

        return material.properties.S[psi]

    @pot("5b", "01")
    def V5b_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * np.dot(q, q) * (material.properties.m_psi[psi]) ** (-2)

        qhat = q / np.linalg.norm(q)

        return C * np.array(
            [L - qhat * np.dot(qhat, L) for L in material.properties.L[psi]]
        )

    @pot("5a", "11")
    def V5a_11(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            N * np.einsum("jki,k->ij", levi_civita, q)
            for N in material.properties.N[psi]
        )

        return V

    @pot("6", "01")
    def V6_01(q, psi, material, m_chi, S_chi):

        C = np.dot(q, q) * (material.properties.m_psi[psi]) ** (-2)

        qhat = q / np.linalg.norm(q)

        return C * np.array(
            [qhat * np.dot(qhat, S) for S in material.properties.S[psi]]
        )

    @pot("7a", "00")
    def V7a_00(q, psi, material, m_chi, S_chi):

        C = -(0.5) * (m_chi) ** (-1.0)

        return C * np.array([np.dot(q, S) for S in material.properties.S[psi]])

    @pot("7b", "00")
    def V7b_00(q, psi, material, m_chi, S_chi):

        C = -(0.5) * (m_chi) ** (-1.0) * 1j

        return C * np.array(
            [
                np.einsum("ijk,ij,k->", levi_civita, LxS, q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @pot("8a", "01")
    def V8a_01(q, psi, material, m_chi, S_chi):

        C = 0.5

        return C * np.array([(-N * (q / m_chi)) for N in material.properties.N[psi]])

    @pot("8a", "11")
    def V8a_11(q, psi, material, m_chi, S_chi):

        C = 1

        return C * np.array([N * np.identity(3) for N in material.properties.N[psi]])

    @pot("8b", "01")
    def V8b_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * (1j / material.properties.m_psi[psi])

        return C * np.array([np.cross(q, L) for L in material.properties.L[psi]])

    @pot("9", "01")
    def V9_01(q, psi, material, m_chi, S_chi):

        C = -(1j / material.properties.m_psi[psi])

        return C * np.array([np.cross(q, S) for S in material.properties.S[psi]])

    @pot("10", "00")
    def V10_00(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array([np.dot(q, S) for S in material.properties.S[psi]])

    @pot("11", "01")
    def V11_01(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array([N * q for N in material.properties.N[psi]])

    @pot("12a", "01")
    def V12a_01(q, psi, material, m_chi, S_chi):

        C = -m_chi / 2

        return C * np.array([np.cross(S, q) for S in material.properties.S[psi]])

    @pot("12a", "11")
    def V12a_11(q, psi, material, m_chi, S_chi):

        C = 1

        return C * np.array(
            [np.einsum("jki,k->ij", levi_civita, S) for S in material.properties.S[psi]]
        )

    @pot("12b", "01")
    def V12b_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * (1j / material.properties.m_psi[psi])

        return C * np.array(
            [
                np.trace(LxS) * q - np.matmul(LxS, q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @pot("13a", "01")
    def V13a_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * (1j / material.properties.m_psi[psi]) / m_chi

        return C * np.array([np.dot(q, S) * q for S in material.properties.S[psi]])

    @pot("13a", "11")
    def V13a_11(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.dot(q, S) * np.identity(3) for S in material.properties.S[psi]]
        )

    @pot("13b", "01")
    def V13b_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.cross(np.matmul(LxS, q), q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @pot("14a", "01")
    def V14a_11(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.einsum("i,j->ij", S, q) for S in material.properties.S[psi]]
        )

    @pot("14b", "01")
    def V14b_01(q, psi, material, m_chi, S_chi):

        C = 0.5 * material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.einsum("ijk,ij,k->", levi_civita, LxS, q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

        return V

    @pot("15a", "11")
    def V15a_11(q, psi, material, m_chi, S_chi):

        C = -material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.dot(q, S) * np.einsum("jki,k->ij", levi_civita, S, q)
                for S in material.properties.S[psi]
            ]
        )

    @pot("15b", "01")
    def V15b_01(q, psi, material, m_chi, S_chi):

        C = -0.5 * 1j * material.properties.m_psi[psi] ** (-3)

        return (
            C
            * np.dot(q, q)
            * np.array(
                [x @ q - q * (q @ x @ q) for x in material.properties.L_tens_S[psi]]
            )
        )
