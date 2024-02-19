"""
    
    Non-relativistic operators

    The scattering potential V can be decomposed as

    V = V^00 + V^11 \cdot S_\chi + V^12 \cdot v + v \cdot (V^20 @ S_\chi)

    with cofficients V^00 (scalar), V^1{1,2} (vectors) and V^{20} (tensor).

    The coefficients are defined in my notes and the rate can be 
    calculated using the formula in my notes.
"""

import numpy as np

from DARK.constants import levi_civita


# TODO: c_dict and c_dict_form should prob just be merged?
class Model:
    def __init__(
        self,
        name: str,
        coeff: dict,
        coeff_qmS: dict,
        Fmed_power: int = 0,
        power_V: int = 0,
        S_chi: float = 0.5,
    ):
        """
        name: string
        Fmed_power: float, negative power of q in the Fmed term
        power_V: float, power of q in the V term (for special mesh)
        s_chi float, spin of DM particle
        """
        self.name = name

        self.Fmed_power = Fmed_power
        self.power_V = power_V
        self.S_chi = S_chi
        self.coeff = coeff
        self.coeff_qmS = coeff_qmS  # Needs better name
        self.operators, self.particles = self.get_operators_and_fermions(coeff)

    def screen_coeff(self, q, m_chi, epsilon):
        """
        Screen the coefficients by the form factor
        """
        q2_qepsq = np.linalg.norm(q, axis=1) ** 2 / np.einsum(
            "ij,jk,ik->i", q, epsilon, q
        )
        screened_coeff_qmS = {alpha: {} for alpha in self.operators}
        for alpha, c_alpha in self.coeff.items():
            screened_c = {
                "e": q2_qepsq * c_alpha.get("e", 0),
                "n": c_alpha.get("n", 0),
                "p": c_alpha.get("p", 0) + (1 - q2_qepsq) * c_alpha.get("e", 0),
            }
            for psi in c_alpha.keys():
                screened_coeff_qmS[alpha][psi] = screened_c[
                    psi
                ] * self.coeff_qmS(alpha, psi, q, m_chi, self.S_chi)

    @staticmethod
    def get_operators_and_fermions(coeff):
        """
        Gets the non-zero operators and particles (psi) from the c_dict
        """

        nonzero_pairs = [
            (alpha, psi)
            for alpha, c_alpha in coeff.items()
            for psi, c_psi in c_alpha.items()
            if c_psi != 0
        ]
        return {pair[0] for pair in nonzero_pairs}, {
            pair[1] for pair in nonzero_pairs
        }


class Potential:
    def __init__(self, model):
        self.operators = model.operators
        self.particles = model.particles
        self.coeff = model.coeff
        self.full_coeff = model.full_coeff

    def eval_V(self, q, material, m_chi, S_chi):
        # TODO: expand this to work for arrays of q (needs to change all the way down)
        full_V = self._get_full_V()
        expansions = set(
            key for alpha in self.operators for key in full_V[alpha].keys()
        )

        # Determine which velocity integrals are needed
        self.needs_g1 = True if "01" or "10" in expansions else False
        self.needs_g2 = True if "11" in expansions else False

        V = {
            exp_id: self.get_zeros(exp_id, material.n_atoms)
            for exp_id in expansions
        }
        # TODO: write this nicer
        for psi in self.particles:
            for alpha in self.operators:
                C = self.coeff[alpha][psi] * self.full_coeff(
                    alpha, psi, q, m_chi, S_chi
                )
                for exp_id, V_func in full_V[alpha].items():
                    # print(f"V^({psi})_{alpha}_{exp_id}")
                    V[exp_id] += C * V_func(q, psi, material, m_chi, S_chi)
        return V

    @classmethod
    def _get_full_V(cls):

        # TODO: I don't like that this requires specifically naming the methods
        # Can this be done with a decorator so that the function name doesn't matter?
        # The decorator would need arguments to specify the operator id and expansion id

        # Get all the methods that start with V
        methods = [
            getattr(cls, method)
            for method in dir(cls)
            if method.startswith("V")
        ]

        # Extract ids from the name (e.g. V3b_10 -> 3b, 10)
        operators = list(method.__name__.split("_") for method in methods)
        operators = [(op[0].lstrip("V"), op[1]) for op in operators]

        V = {op_id: {} for op_id in set(op[0] for op in operators)}
        for op_id, exp_id in operators:
            V[op_id][exp_id] = methods.pop(0)
        return V

    @staticmethod
    def get_zeros(exp_id, n_atoms):
        if exp_id == "00":
            return np.zeros(n_atoms, dtype=complex)
        elif exp_id == "01" or exp_id == "10":
            return np.zeros((n_atoms, 3), dtype=complex)
        elif exp_id == "11":
            return np.zeros((n_atoms, 3, 3), dtype=complex)

    # TODO: Rewrite all of the Vs in terms of qhat
    # TODO: add option to specify L \times S and/or L \dot S in input
    #       and reconstruct the relevant portions of L \otimes S form them?
    @staticmethod
    def V1_00(q, psi, material, m_chi, S_chi):

        return material.properties.N[psi]

    @staticmethod
    def V3b_00(q, psi, material, m_chi, S_chi):

        V = np.zeros(material.n_atoms, dtype=complex)

        qhat = q / np.linalg.norm(q)

        C = -0.5 * np.linalg.norm(q) ** 2 / material.properties.m_psi[psi] ** 2

        for j in range(material.n_atoms):

            LxS_V = material.properties.L_tens_S[psi][j]

            V[j] = C * (np.trace(LxS_V) - np.dot(qhat, np.matmul(LxS_V, qhat)))

        return V

    @staticmethod
    def V3a_12(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return np.array(
            [C * np.cross(Sj, q) for Sj in material.properties.S[psi]]
        )

    @staticmethod
    def V4_11(q, psi, material, m_chi, S_chi):

        return material.properties.S[psi]

    @staticmethod
    def V5b_11(q, psi, material, m_chi, S_chi):

        C = -0.5 * np.dot(q, q) * (material.properties.m_psi[psi]) ** (-2)

        qhat = q / np.linalg.norm(q)

        return C * np.array(
            [L - qhat * np.dot(qhat, L) for L in material.properties.L[psi]]
        )

    @staticmethod
    def V5a_20(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [
                N * np.einsum("jki,k->ij", levi_civita, q)
                for N in material.properties.N[psi]
            ]
        )

    @staticmethod
    def V6_11(q, psi, material, m_chi, S_chi):
        C = np.dot(q, q) * (material.properties.m_psi[psi]) ** (-2)

        qhat = q / np.linalg.norm(q)

        return C * np.array(
            [qhat * np.dot(qhat, S) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V7a_00(q, psi, material, m_chi, S_chi):
        C = -(0.5) * (m_chi) ** (-1.0)

        return C * np.array([np.dot(q, S) for S in material.properties.S[psi]])

    @staticmethod
    def V7b_00(q, psi, material, m_chi, S_chi):

        C = -(0.5) * (m_chi) ** (-1.0) * 1j

        return C * np.array(
            [
                np.einsum("ijk,ij,k->", levi_civita, LxS, q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V8a_11(q, psi, material, m_chi, S_chi):

        C = -q / 2 / m_chi

        return C * np.array([N for N in material.properties.N[psi]])

    @staticmethod
    def V8a_12(q, psi, material, m_chi, S_chi):

        C = 1

        return C * np.array(
            [N * np.identity(3) for N in material.properties.N[psi]]
        )

    @staticmethod
    def V8b_01(q, psi, material, m_chi, S_chi):
        C = 0.5 * (1j / material.properties.m_psi[psi])

        return C * np.array(
            [np.cross(L, q) for L in material.properties.L[psi]]
        )

    @staticmethod
    def V9_11(q, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.cross(S, q) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V10_00(q, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.array([np.dot(q, S) for S in material.properties.S[psi]])

    @staticmethod
    def V11_11(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array([N * q for N in material.properties.N[psi]])

    @staticmethod
    def V12a_01(q, psi, material, m_chi, S_chi):

        C = -m_chi / 2

        return C * np.array(
            [np.cross(S, q) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V12a_20(q, psi, material, m_chi, S_chi):

        C = 1

        return C * np.array(
            [
                np.einsum("jki,k->ij", levi_civita, S)
                for S in material.properties.S[psi]
            ]
        )

    @staticmethod
    def V12b_11(q, psi, material, m_chi, S_chi):

        C = -0.5 * (1j / material.properties.m_psi[psi])

        return C * np.array(
            [
                np.trace(LxS) * q - np.matmul(LxS, q)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V13a_11(q, psi, material, m_chi, S_chi):

        C = -0.5 * (1j / material.properties.m_psi[psi]) / m_chi

        return C * np.array(
            [np.dot(q, S) * q for S in material.properties.S[psi]]
        )

    @staticmethod
    def V13a_20(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.dot(q, S) * np.identity(3) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V13b_11(q, psi, material, m_chi, S_chi):

        C = 0.5 * material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.cross(q, np.matmul(LxS, q))
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V14a_11(q, psi, material, m_chi, S_chi):

        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.einsum("i,j->ij", S, q) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V14b_11(q, psi, material, m_chi, S_chi):
        C = 0.5 * material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.einsum("ijk,ij,k->", levi_civita, LxS, q) * q
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V15a_20(q, psi, material, m_chi, S_chi):
        C = -material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.dot(q, S) * np.einsum("jki,k->ij", levi_civita, q)
                for S in material.properties.S[psi]
            ]
        )

    @staticmethod
    def V15b_11(q, psi, material, m_chi, S_chi):

        C = -0.5 * 1j * material.properties.m_psi[psi] ** (-3)

        return (
            C
            * np.dot(q, q)
            * np.array(
                [
                    LxS @ q - q * (q @ LxS @ q)
                    for LxS in material.properties.L_tens_S[psi]
                ]
            )
        )
