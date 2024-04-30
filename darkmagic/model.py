import warnings
from typing import Callable

import numpy as np

from darkmagic.constants import levi_civita
from darkmagic.numerics import SphericalGrid

SUPPORTED_OPERATORS = {
    "1",
    "3a",
    "3b",
    "4",
    "5a",
    "5b",
    "6",
    "7a",
    "7b",
    "8a",
    "8b",
    "9",
    "10",
    "11",
    "12a",
    "12b",
    "13a",
    "13b",
    "14a",
    "14b",
    "15a",
    "15b",
}


# TODO: c_dict and c_dict_form should prob just be merged?
class Model:
    def __init__(
        self,
        name: str,
        coeff_prefactor: dict,
        coeff_func: dict,
        F_med_prop: Callable[[SphericalGrid], np.array] | None = None,
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

        self.F_med_prop_prop_val = 0  # temporary

        self.power_V = power_V
        self.S_chi = S_chi
        self.coeff_prefactor = coeff_prefactor
        self.coeff_func = coeff_func
        self.operators, self.particles = self._get_operators_and_particles()
        if F_med_prop is None:

            def ones(grid):
                return np.ones_like(grid.q_norm)

            F_med_prop = ones
        self.F_med_prop = F_med_prop
        self._validate_coefficients()

    def get_unscreened_coeff(self, alpha, psi, grid, m_chi, S_chi):
        return self.coeff_prefactor[alpha][psi] * self.coeff_func[alpha][psi](
            grid, m_chi, S_chi
        )

    def compute_screened_coeff(self, grid, epsilon, m_chi, S_chi):
        """
        Screen the coefficients by the form factor
        """
        q_eps_q = np.sum(grid.qhat_qhat * epsilon[None, :], axis=(-1, -2))
        screened_coeff = {
            alpha: {psi: self.coeff_prefactor[alpha][psi] for psi in self.particles}
            for alpha in self.operators
        }
        for alpha, c_alpha in screened_coeff.items():
            for psi in c_alpha.keys():
                if psi == "e":
                    c_alpha[psi] *= 1 / q_eps_q
                elif psi == "p":
                    # TODO: is this correct?
                    c_alpha[psi] += (1 - 1 / q_eps_q) * c_alpha.get("e", 0)
                elif psi == "n":
                    c_alpha[psi] *= np.ones_like(q_eps_q)
                # neutrons are unscreened
                c_alpha[psi] *= self.coeff_func[alpha][psi](grid, m_chi, S_chi)

        return screened_coeff

    def _get_operators_and_particles(self):
        """
        Gets the non-zero operators and particles (psi) from the coefficients
        """

        nonzero_pairs = [
            (alpha, psi)
            for alpha, c_alpha in self.coeff_prefactor.items()
            for psi, c_psi in c_alpha.items()
            if c_psi != 0
        ]
        operators = {pair[0] for pair in nonzero_pairs}
        particles = {pair[1] for pair in nonzero_pairs}

        return operators, particles

    def _validate_coefficients(self):
        """
        Validates the coefficients. Namely, this means that the coefficient functions
        are defined for every (particle,operator) pair with a non-zero coefficient prefactor, and that every operator is supported.

        Raises:
            MissingCoefficientFunctionException: If an operator has a non-zero coefficient prefactor but no coefficient function.
            UnsupportedOperatorException: If an operator is not supported.

        Warns:
            ExtraCoefficientFunctionWarning: If an operator has a coefficient function but no non-zero coefficient prefactor, so the operator will be ignored.
        """
        # Check that every operator is supported
        for alpha in self.operators:
            if alpha not in SUPPORTED_OPERATORS:
                raise UnsupportedOperatorException(
                    f"Operator {alpha} is not supported."
                )
        # Ignore operators that have coefficient function defined but
        # no non-zero coefficient prefactor
        for alpha in list(self.coeff_func.keys()):
            if alpha not in self.operators:
                warnings.warn(
                    f"Operator {alpha} has a coefficient function defined but no "
                    "corresponding nonzero coefficient prefactor for any particle. "
                    "It will be ignored.",
                    ExtraCoefficientFunctionWarning,
                )
                del self.coeff_func[alpha]
                continue
            for psi in list(self.coeff_func[alpha].keys()):
                if psi not in self.particles:
                    warnings.warn(
                        f"Operator {alpha} has a coefficient function defined for "
                        f"particle {psi} but no corresponding nonzero coefficient "
                        "prefactor. It will be ignored.",
                        ExtraCoefficientFunctionWarning,
                    )
                    del self.coeff_func[alpha][psi]
        # Check that every operator with a non-zero coefficient prefactor has
        # a coefficient function defined
        if self.operators != set(self.coeff_func.keys()):
            raise MissingCoefficientFunctionException(
                "Some operators with non-zero coefficient prefactors have "
                "no coefficient functions defined."
            )
        particles_from_func = {
            key for alpha in self.coeff_func for key in self.coeff_func[alpha].keys()
        }
        if self.particles != particles_from_func:
            raise MissingCoefficientFunctionException(
                "Some particles with non-zero coefficient prefactors "
                "have no coefficient functions defined."
            )


class Potential:
    def __init__(self, model):
        self.operators = model.operators
        self.particles = model.particles
        self.c = model.compute_screened_coeff

    def eval_V(self, grid, material, m_chi, S_chi):
        """
        Evaluate the full potential V_j(q) for the given grid and material
        """
        full_V = self._get_full_V()
        terms = {key for alpha in self.operators for key in full_V[alpha].keys()}

        # Determine which velocity integrals are needed
        self.needs_g1 = bool("12" or "11" in terms)
        self.needs_g2 = "20" in terms

        # TODO: Should be merged with _get_zeros?
        V = {t: self._get_zeros(t, material.n_atoms, grid) for t in terms}

        def get_slice(t):
            return (slice(None),) + (None,) * (V[t].ndim - 1)

        coeff = self.c(grid, material.epsilon, m_chi, S_chi)
        # TODO: write this nicer
        for psi in self.particles:
            for alpha in self.operators:
                C = coeff[alpha][psi]
                for t, V_func in full_V[alpha].items():
                    # print(f"V^({psi})_{alpha}_{t}")
                    # print(V[t].shape)
                    # print(V_func(grid, psi, material, m_chi, S_chi).shape)
                    # print(C.shape)
                    V[t] += C[get_slice(t)] * V_func(grid, psi, material, m_chi, S_chi)
        return V

    @classmethod
    def _get_full_V(cls):
        # TODO: I don't like that this requires specifically naming the methods
        # Can this be done with a decorator so that the function name doesn't matter?
        # The decorator would need arguments to specify the operator id and expansion id

        # Get all the methods that start with V
        methods = [
            getattr(cls, method) for method in dir(cls) if method.startswith("V")
        ]

        # Extract term type from the name (e.g. V3b_10 -> 3b, 10)
        operators = [method.__name__.split("_") for method in methods]
        operators = [(op[0].lstrip("V"), op[1]) for op in operators]

        V = {op_id: {} for op_id in {op[0] for op in operators}}
        for op_id, exp_id in operators:
            V[op_id][exp_id] = methods.pop(0)
        return V

    @staticmethod
    def _get_zeros(term, n_atoms, grid):
        nq = len(grid.q_cart)
        if term == "00":
            return np.zeros((nq, n_atoms), dtype=complex)
        elif term in ["11", "12"]:
            return np.zeros((nq, n_atoms, 3), dtype=complex)
        elif term == "20":
            return np.zeros((nq, n_atoms, 3, 3), dtype=complex)
        else:
            raise ValueError(f"Unknown term type {term}")

    # TODO: Rewrite all of the Vs in terms of qhat
    # TODO: add option to specify L \times S and/or L \dot S in input
    #       and reconstruct the relevant portions of L \otimes S form them?
    @staticmethod
    def V1_00(grid, psi, material, m_chi, S_chi):
        # NOTE: ported to grid
        return material.properties.N[psi] * np.ones(
            (grid.q_cart.shape[0], material.n_atoms)
        )

    @staticmethod
    def V3b_00(grid, psi, material, m_chi, S_chi):
        V = np.zeros(material.n_atoms, dtype=complex)

        qhat = grid.q_hat

        C = -0.5 * grid.q_norm**2 / material.properties.m_psi[psi] ** 2

        for j in range(material.n_atoms):
            LxS_V = material.properties.L_tens_S[psi][j]

            V[j] = C * (np.trace(LxS_V) - np.dot(qhat, np.matmul(LxS_V, qhat)))

        return V

    @staticmethod
    def V3a_12(grid, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.cross(material.properties.S[psi], grid.q_cart)
        # return np.array(
        #    [C * np.cross(Sj, grid.q_cart) for Sj in material.properties.S[psi]]
        # )

    @staticmethod
    def V4_11(grid, psi, material, m_chi, S_chi):
        return material.properties.S[psi][:, None, None] * np.ones(
            (grid.q_cart.shape[0], 1)
        )

    @staticmethod
    def V5b_11(grid, psi, material, m_chi, S_chi):
        C = -0.5 * grid.q_norm**2 * (material.properties.m_psi[psi]) ** (-2)

        qhat = grid.q_hat

        return C * np.array(
            [L - qhat * np.dot(qhat, L) for L in material.properties.L[psi]]
        )

    @staticmethod
    def V5a_20(grid, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [
                N * np.einsum("jki,k->ij", levi_civita, grid.q_cart)
                for N in material.properties.N[psi]
            ]
        )

    @staticmethod
    def V6_11(grid, psi, material, m_chi, S_chi):
        C = grid.q_norm**2 * (material.properties.m_psi[psi]) ** (-2)

        qhat = grid.q_hat

        return C * np.array(
            [qhat * np.dot(qhat, S) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V7a_00(grid, psi, material, m_chi, S_chi):
        C = -(0.5) * (m_chi) ** (-1.0)

        return C * np.array(
            [np.dot(grid.q_cart, S) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V7b_00(grid, psi, material, m_chi, S_chi):
        C = -(0.5) * (m_chi) ** (-1.0) * 1j

        return C * np.array(
            [
                np.einsum("ijk,ij,kl->k", levi_civita, LxS, grid.q_cart)
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V8a_11(grid, psi, material, m_chi, S_chi):
        C = -grid.q_cart / 2 / m_chi

        return C * np.array(list(material.properties.N[psi]))[:, None]

    @staticmethod
    def V8a_20(grid, psi, material, m_chi, S_chi):
        C = 1

        return C * np.array([N * np.identity(3) for N in material.properties.N[psi]])

    @staticmethod
    def V8b_11(grid, psi, material, m_chi, S_chi):
        C = 0.5 * (1j / material.properties.m_psi[psi])

        return C * np.array(
            [np.cross(L, grid.q_cart) for L in material.properties.L[psi]]
        )

    @staticmethod
    def V9_11(grid, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.cross(S, grid.q_cart) for S in material.properties.S[psi]]
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
    def V12a_11(q, psi, material, m_chi, S_chi):
        C = -m_chi / 2

        return C * np.array([np.cross(S, q) for S in material.properties.S[psi]])

    @staticmethod
    def V12a_20(q, psi, material, m_chi, S_chi):
        C = 1

        return C * np.array(
            [np.einsum("jki,k->ij", levi_civita, S) for S in material.properties.S[psi]]
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

        return C * np.array([np.dot(q, S) * q for S in material.properties.S[psi]])

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
    def V14a_11(grid, psi, material, m_chi, S_chi):
        C = -1j / (2 * material.properties.m_psi[psi] * m_chi)

        return C * np.array(
            [
                np.einsum("i,j,k->i", grid.q_cart, S, grid.q_cart)
                for S in material.properties.S[psi]
            ]
        )

    @staticmethod
    def V14a_20(grid, psi, material, m_chi, S_chi):
        C = 1j / material.properties.m_psi[psi]

        return C * np.array(
            [np.einsum("i,j->ij", S, grid.q_cart) for S in material.properties.S[psi]]
        )

    @staticmethod
    def V14b_11(grid, psi, material, m_chi, S_chi):
        C = 0.5 * material.properties.m_psi[psi] ** (-2)

        return C * np.array(
            [
                np.einsum("ijk,ij,k->", levi_civita, LxS, grid.q_cart) * grid.q_cart
                for LxS in material.properties.L_tens_S[psi]
            ]
        )

    @staticmethod
    def V15a_20(grid, psi, material, m_chi, S_chi):
        C = -(material.properties.m_psi[psi] ** (-2))

        return C * np.array(
            [
                np.dot(grid.q_cart, S)
                * np.einsum("jki,k->ij", levi_civita, grid.q_cart)
                for S in material.properties.S[psi]
            ]
        )

    @staticmethod
    def V15b_11(grid, psi, material, m_chi, S_chi):
        C = -0.5 * 1j * material.properties.m_psi[psi] ** (-3)

        # LxS = material.properties.L_tens_S[psi]
        # M1 = (LxS @ grid.q_cart[..., None]).squeeze()
        return (
            C
            * grid.q_norm**2
            * np.array(
                [
                    LxS @ grid.q_cart - grid.q_cart * (grid.q_cart @ LxS @ grid.q_cart)
                    for LxS in material.properties.L_tens_S[psi]
                ]
            )
        )


class MissingCoefficientFunctionException(Exception):
    """
    Raised when an operator has a non-zero coefficient prefactor but no coefficient function.
    """

    pass


class ExtraCoefficientFunctionWarning(Warning):
    """
    Warning that an operator has a coefficient function but no non-zero coefficient prefactor, so the operator will be ignored.
    """

    pass


class UnsupportedOperatorException(Exception):
    """
    Raised when an operator is not supported.
    """

    pass
