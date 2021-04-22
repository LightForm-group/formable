"""`formable.yielding.yield_functions.py`

Yield function definitions (represented here as Python classes) from the literature.

Notes
-----
Each yield function class must define two methods:
    __init__
        All yield function parameters are assigned as attributes.
    residual
        Static method used as fitting callable, and by the parent class method
        `get_value`. This method must be decorated by `yield_function_fitter`, which
        ensures that all parameters are contained with the `**kwargs` dict argument,
        regardless of those that are passed in the `fitting_params` list argument. Within
        this method, all parameters should be accessed from the `**kwargs` dict (not from
        the `fitting_params` list).

"""

import numpy as np

from formable import maths_utils
from formable.yielding import YieldFunction, yield_function_fitter


class VonMises(YieldFunction):

    PARAMETERS = [
        'equivalent_stress',
    ]

    def __init__(self, equivalent_stress):

        self.equivalent_stress = equivalent_stress

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        princ = np.sort(np.linalg.eigvals(stress_states), axis=-1)[:, ::-1]

        diff_sq_sum = (
            (princ[:, 0] - princ[:, 1])**2 +
            (princ[:, 0] - princ[:, 2])**2 +
            (princ[:, 1] - princ[:, 2])**2
        )

        value = (np.sqrt(diff_sq_sum / 2) / kwargs['equivalent_stress']) - 1

        return value


class Tresca(YieldFunction):

    PARAMETERS = [
        'equivalent_stress',
    ]

    def __init__(self, equivalent_stress):

        self.equivalent_stress = equivalent_stress

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        princ = np.sort(np.linalg.eigvals(stress_states), axis=-1)[:, ::-1]

        diff = np.array([
            (princ[:, 0] - princ[:, 1]),
            (princ[:, 0] - princ[:, 2]),
            (princ[:, 1] - princ[:, 2]),
        ])
        value = (np.max(diff, axis=0) / kwargs['equivalent_stress']) - 1

        return value


class Barlat_Yld91(YieldFunction):
    """Barlat "Yld91" 1991 yield criterion for anisotropic plasticity.

    Notes
    -----
    Should reduce to Hill 1948 for `exponent=2`

    """

    PARAMETERS = [
        'a',
        'b',
        'c',
        'f',
        'g',
        'h',
        'equivalent_stress',
        'exponent',
    ]

    LITERATURE_VALUES = {
        'AA3014_recrystallised_3D': {
            'Zhang (2016)': {
                'doi': '10.1016/j.ijplas.2016.01.002',
                'notes': 'In Eq. 18, Term "+3π" should be "-3π" according to original Barlat paper.',
                'parameters': {
                    'a': 0.9976,
                    'b': 1.0615,
                    'c': 1.0315,
                    'f': 0.9759,
                    'g': 0.9525,
                    'h': 0.9627,
                    'exponent': 5.3041,
                }
            }
        }
    }

    def __init__(self, a, b, c, f, g, h, equivalent_stress, exponent):

        self.a = a
        self.b = b
        self.c = c
        self.f = f
        self.g = g
        self.h = h
        self.equivalent_stress = equivalent_stress
        self.exponent = exponent

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        f, g, h = kwargs['f'], kwargs['g'], kwargs['h']
        exponent = kwargs['exponent']
        equivalent_stress = kwargs['equivalent_stress']

        # Following notation in original paper:
        A = stress_states[:, 1, 1] - stress_states[:, 2, 2]
        B = stress_states[:, 2, 2] - stress_states[:, 0, 0]
        C = stress_states[:, 0, 0] - stress_states[:, 1, 1]
        F = stress_states[:, 1, 2]
        G = stress_states[:, 2, 0]
        H = stress_states[:, 0, 1]

        aA = a * A
        bB = b * B
        cC = c * C
        fF = f * F
        gG = g * G
        hH = h * H

        # Stress deviator invariants:
        I_2 = (
            ((fF**2 + gG**2 + hH**2) / 3) +
            (((aA - cC)**2 + (cC - bB)**2 + (bB - aA)**2) / 54)
        )
        I_3 = (
            ((cC - bB) * (aA - cC) * (bB - aA) / 54) +
            (fF * gG * hH) - (
                (
                    ((cC - bB) * fF**2) +
                    ((aA - cC) * gG**2) +
                    ((bB - aA) * hH**2)
                ) / 6
            )
        )

        with np.errstate(invalid='ignore'):
            # For hydrostatic stresses, I_2 will be zero, so theta will be np.nan:
            theta = np.arccos(I_3 / I_2**(3/2))

        finite_idx = np.isfinite(theta)

        Phi_1 = (+2 * np.cos(((2 * theta[finite_idx]) + np.pi) / 6))**exponent
        Phi_2 = (+2 * np.cos(((2 * theta[finite_idx]) - (3 * np.pi)) / 6))**exponent
        Phi_3 = (-2 * np.cos(((2 * theta[finite_idx]) + (5 * np.pi)) / 6))**exponent

        # Where theta is np.nan, Phi is zero:
        Phi = np.zeros(stress_states.shape[0])
        Phi[finite_idx] = (
            ((3 * I_2[finite_idx])**(exponent / 2)) * (Phi_1 + Phi_2 + Phi_3)
        )

        value = (((Phi / 2)**(1 / exponent)) / equivalent_stress) - 1

        return value


class Barlat_Yld2000_2D(YieldFunction):
    """Barlat "Yld2000-2D" yield criterion for plane stress anisotropic plasticity.

    Notes
    -----
    From Ref. [1].

    References
    ----------
    [1]: Barlat, F. et al. "Plane Stress Yield Function for Aluminum Alloy Sheets—Part 1: Theory".
         International Journal of Plasticity 19, no. 9 (1 September 2003): 1297–1319.
         https://doi.org/10.1016/S0749-6419(02)00019-0.

    """

    PARAMETERS = [
        'a1',
        'a2',
        'a3',
        'a4',
        'a5',
        'a6',
        'a7',
        'a8',
        'equivalent_stress',
        'exponent',
    ]

    LITERATURE_VALUES = {
        'AA3014_recrystallised': {
            'Zhang (2016)': {
                'doi': '10.1016/j.ijplas.2016.01.002',
                'notes': '',
                'parameters': {
                    'a1': 0.9349,
                    'a2': 1.0671,
                    'a3': 0.9690,
                    'a4': 0.9861,
                    'a5': 1.0452,
                    'a6': 0.9999,
                    'a7': 0.9097,
                    'a8': 1.0805,
                    'exponent': 5.9992,
                }
            }
        }
    }

    def __init__(self, a1, a2, a3, a4, a5, a6, a7, a8, equivalent_stress, exponent):

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7
        self.a8 = a8
        self.equivalent_stress = equivalent_stress
        self.exponent = exponent

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        a1, a2, a3, a4 = kwargs['a1'], kwargs['a2'], kwargs['a3'], kwargs['a4']
        a5, a6, a7, a8 = kwargs['a5'], kwargs['a6'], kwargs['a7'], kwargs['a8']
        exponent = kwargs['exponent']
        equivalent_stress = kwargs['equivalent_stress']

        stress_v = maths_utils.to_symmetric_voigt_notation(stress_states)

        # Select only the planar stress components:
        planar_stress = stress_v[:, [0, 1, 5]]
        stress_cols = planar_stress.T  # shape (3, N)

        L_prime = np.array([
            [(2 * a1), -a1, 0],
            [-a2, (2 * a2), 0],
            [0, 0, (3 * a7)],
        ]) / 3

        L_dprime = np.array([
            [
                ((8 * a5) - (2 * a3) - (2 * a6) + (2 * a4)),
                ((4 * a6) - (4 * a4) - (4 * a5) + a3),
                0,
            ],
            [
                ((4 * a3) - (4 * a5) - (4 * a4) + a6),
                ((8 * a4) - (2 * a6) - (2 * a3) + (2 * a5)),
                0,
            ],
            [
                0,
                0,
                (9 * a8),
            ],
        ]) / 9

        X_prime = L_prime @ stress_cols
        X_dprime = L_dprime @ stress_cols

        X_prime_ps = maths_utils.get_plane_stress_principle_stresses(X_prime)
        X_dprime_ps = maths_utils.get_plane_stress_principle_stresses(X_dprime)

        phi_prime = np.abs(X_prime_ps[0] - X_prime_ps[1]) ** exponent
        phi_dprime = (
            np.abs((2 * X_dprime_ps[1]) + X_dprime_ps[0]) ** exponent +
            np.abs((2 * X_dprime_ps[0]) + X_dprime_ps[1]) ** exponent
        )

        phi = phi_prime + phi_dprime

        value = (((phi / 2)**(1 / exponent)) / equivalent_stress) - 1

        return value


class Barlat_Yld2004_18p(YieldFunction):
    """Barlat "Yld2004-18p" yield criterion for anisotropic plasticity.

    Notes
    -----
    Yield criterion defined with eighteen anisotropy parameters following
    Ref. [1].

    References
    ----------
    [1] Barlat, F. et al. 'Linear Transfomation-Based Anisotropic Yield Functions'.
        International Journal of Plasticity 21, no. 5 (1 May 2005): 1009–39.
        https://doi.org/10.1016/j.ijplas.2004.06.004.

    """

    PARAMETERS = [
        'c_p_12',
        'c_p_21',
        'c_p_23',
        'c_p_32',
        'c_p_31',
        'c_p_13',
        'c_p_44',
        'c_p_55',
        'c_p_66',
        'c_dp_12',
        'c_dp_21',
        'c_dp_23',
        'c_dp_32',
        'c_dp_31',
        'c_dp_13',
        'c_dp_44',
        'c_dp_55',
        'c_dp_66',
        'equivalent_stress',
        'exponent',
    ]

    FORMATTED_PARAMETER_NAMES = {
        'unicode': {
            'c_p_12':  'c′₁₂',
            'c_p_21':  'c′₂₁',
            'c_p_23':  'c′₂₃',
            'c_p_32':  'c′₃₂',
            'c_p_31':  'c′₃₁',
            'c_p_13':  'c′₁₃',
            'c_p_44':  'c′₄₄',
            'c_p_55':  'c′₅₅',
            'c_p_66':  'c′₆₆',
            'c_dp_12': 'c″₁₂',
            'c_dp_21': 'c″₂₁',
            'c_dp_23': 'c″₂₃',
            'c_dp_32': 'c″₃₂',
            'c_dp_31': 'c″₃₁',
            'c_dp_13': 'c″₁₃',
            'c_dp_44': 'c″₄₄',
            'c_dp_55': 'c″₅₅',
            'c_dp_66': 'c″₆₆',
        },
    }

    LITERATURE_VALUES = {
        'AA3014_recrystallised_3D': {
            'Zhang (2016)': {
                'doi': '10.1016/j.ijplas.2016.01.002',
                'notes': '"c_dp_" parameters are listed as "d_" paramaters in this work.',
                'parameters': {
                    'c_p_12':   1.1533,
                    'c_p_21':   1.6206,
                    'c_p_23':   0.9688,
                    'c_p_32':   0.6850,
                    'c_p_31':   0.9189,
                    'c_p_13':   1.4709,
                    'c_p_44':   1.4253,
                    'c_p_55':   0.7399,
                    'c_p_66':   1.1367,
                    'c_dp_12': -0.3329,
                    'c_dp_21':  1.0360,
                    'c_dp_23':  1.4072,
                    'c_dp_32':  0.7698,
                    'c_dp_31':  0.8263,
                    'c_dp_13':  0.4204,
                    'c_dp_44':  0.2457,
                    'c_dp_55':  1.1198,
                    'c_dp_66': -0.6515,
                    'exponent': 6.6847,
                }
            }
        }
    }

    def __init__(self, c_p_12, c_p_21, c_p_23, c_p_32, c_p_31, c_p_13, c_p_44, c_p_55,
                 c_p_66, c_dp_12, c_dp_21, c_dp_23, c_dp_32, c_dp_31, c_dp_13, c_dp_44,
                 c_dp_55, c_dp_66, equivalent_stress, exponent):

        self.c_p_12 = c_p_12
        self.c_p_21 = c_p_21
        self.c_p_23 = c_p_23
        self.c_p_32 = c_p_32
        self.c_p_31 = c_p_31
        self.c_p_13 = c_p_13
        self.c_p_44 = c_p_44
        self.c_p_55 = c_p_55
        self.c_p_66 = c_p_66

        self.c_dp_12 = c_dp_12
        self.c_dp_21 = c_dp_21
        self.c_dp_23 = c_dp_23
        self.c_dp_32 = c_dp_32
        self.c_dp_31 = c_dp_31
        self.c_dp_13 = c_dp_13
        self.c_dp_44 = c_dp_44
        self.c_dp_55 = c_dp_55
        self.c_dp_66 = c_dp_66

        self.equivalent_stress = equivalent_stress
        self.exponent = exponent

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):
        """
        Parameters
        ----------
        stress_states : ndarray of shape (N, 3, 3)
            N stress states to be used as input experimental/simulated data to fit
            the yield function parameters to.

        """

        def transform_stress(stress, params):
            """Transform stress states using parameters components of a linear
            transformation tensor.

            Parameters
            ----------
            stress : ndarray of shape (N, 6)
            params : list
                List of nine anisotropy parameters composing the linear transformation.

            """

            stress_trans = np.array([
                (stress[:, 1] * -params[0]) + (stress[:, 2] * -params[5]),
                (stress[:, 0] * -params[1]) + (stress[:, 2] * -params[2]),
                (stress[:, 0] * -params[4]) + (stress[:, 1] * -params[3]),
                (stress[:, 3] * params[6]),
                (stress[:, 4] * params[7]),
                (stress[:, 5] * params[8]),
            ]).T

            return stress_trans

        exponent = kwargs['exponent']
        equivalent_stress = kwargs['equivalent_stress']
        params = [
            kwargs['c_p_12'],
            kwargs['c_p_21'],
            kwargs['c_p_23'],
            kwargs['c_p_32'],
            kwargs['c_p_31'],
            kwargs['c_p_13'],
            kwargs['c_p_44'],
            kwargs['c_p_55'],
            kwargs['c_p_66'],
            kwargs['c_dp_12'],
            kwargs['c_dp_21'],
            kwargs['c_dp_23'],
            kwargs['c_dp_32'],
            kwargs['c_dp_31'],
            kwargs['c_dp_13'],
            kwargs['c_dp_44'],
            kwargs['c_dp_55'],
            kwargs['c_dp_66'],
        ]

        stress_v = maths_utils.to_symmetric_voigt_notation(stress_states)
        stress_dev = maths_utils.get_deviatoric_stress(stress_v, voigt=True)

        stress_trans_1 = transform_stress(stress_dev, params[0:9])
        stress_trans_2 = transform_stress(stress_dev, params[9:18])

        # Find principal stress values:
        stress_mat_1 = maths_utils.from_voigt_notation(stress_trans_1)
        stress_mat_2 = maths_utils.from_voigt_notation(stress_trans_2)
        stress_princ_1 = np.sort(np.linalg.eigvals(stress_mat_1), axis=-1)[:, ::-1]
        stress_princ_2 = np.sort(np.linalg.eigvals(stress_mat_2), axis=-1)[:, ::-1]

        princ_sum_terms = [
            np.abs(stress_princ_1[:, i] - stress_princ_2[:, j]) ** exponent
            for i in range(3)
            for j in range(3)
        ]
        princ_sum_terms = np.array(princ_sum_terms).T

        princ_sum = np.sum(princ_sum_terms, axis=-1)
        value = (((princ_sum / 4) ** (1 / exponent)) / equivalent_stress) - 1

        return value


class Hosford(YieldFunction):
    'Hosford yield criterion for isotropic plasticity.'

    PARAMETERS = [
        'equivalent_stress',
        'exponent',
    ]

    def __init__(self, equivalent_stress, exponent):

        self.equivalent_stress = equivalent_stress
        self.exponent = exponent

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        exponent = kwargs['exponent']
        equivalent_stress = kwargs['equivalent_stress']

        princ = np.sort(np.linalg.eigvals(stress_states), axis=-1)[:, ::-1]
        diff_sq_sum = (
            (princ[:, 0] - princ[:, 1])**exponent +
            (princ[:, 0] - princ[:, 2])**exponent +
            (princ[:, 1] - princ[:, 2])**exponent
        )

        value = (((diff_sq_sum / 2)**(1 / exponent)) / equivalent_stress) - 1

        return value


class Hill1979(YieldFunction):
    """Hill 1979 non-quadratic yield criterion for anisotropic plasticity.

    Notes
    -----
    From Eq. 4.5 in Ref. [1], the yield criterion is defined:
    (f(σ_1 - σ_1)^m +
     g(σ_1 - σ_1)^m +
     h(σ_1 - σ_1)^m +
     a(σ_1 - σ_1)^m +
     b(σ_1 - σ_1)^m +
     c(σ_1 - σ_1)^m ) = σ^m

    where f,g,h,a,b,c are fitting parameters, m is the exponent, and σ is the equivalent
    stress (usually taken to be the yield stress in unaxial tension along the rolling
    direction).


    References
    ----------
    [1]: Hill, R. "Theoretical Plasticity of Textured Aggregates". Mathematical Proceedings
         of the Cambridge Philosophical Society 85, no. 1 (January 1979): 179–91.
         https://doi.org/10.1017/S0305004100055596.

    """

    PARAMETERS = [
        'f',
        'g',
        'h',
        'a',
        'b',
        'c',
        'equivalent_stress',
        'exponent',
    ]

    def __init__(self, f, g, h, a, b, c, equivalent_stress, exponent):

        self.f = f
        self.g = g
        self.h = h
        self.a = a
        self.b = b
        self.c = c
        self.equivalent_stress = equivalent_stress
        self.exponent = exponent

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        f, g, h = kwargs['f'], kwargs['g'], kwargs['h']
        a, b, c = kwargs['a'], kwargs['b'], kwargs['c']
        exponent = kwargs['exponent']
        equivalent_stress = kwargs['equivalent_stress']

        princ = np.sort(np.linalg.eigvals(stress_states), axis=-1)[:, ::-1]

        diff_exp_sum = (
            f * np.abs(princ[:, 1] - princ[:, 2])**exponent +
            g * np.abs(princ[:, 2] - princ[:, 0])**exponent +
            h * np.abs(princ[:, 0] - princ[:, 1])**exponent +
            a * np.abs((2 * princ[:, 0]) - princ[:, 1] - princ[:, 2])**exponent +
            b * np.abs((2 * princ[:, 1]) - princ[:, 2] - princ[:, 0])**exponent +
            c * np.abs((2 * princ[:, 2]) - princ[:, 0] - princ[:, 1])**exponent
        )

        value = ((diff_exp_sum ** (1 / exponent)) / equivalent_stress) - 1

        return value


class Hill1948(YieldFunction):
    """Hill 1948 quadratic yield criterion for anisotropic plasticity.

    TODO: checks:  When F=Q=H=L/3=M/3=N/3, should reduce to Mises. [https://core.ac.uk/reader/81085010]
    """

    PARAMETERS = [
        'F',
        'G',
        'H',
        'L',
        'M',
        'N',
        'equivalent_stress',
    ]

    LITERATURE_VALUES = {
        'TA-6V': {
            'Giles (2012)': {
                'doi': '10.1016/j.piutam.2012.03.008',
                'notes': 'Calculated from experimentally determined R-values.',
                'parameters': {
                    'F': 0.4803,
                    'G': 0.9337,
                    'H': 1.0663,
                    'L': 0,
                    'M': 0,
                    'N': 3.9804,
                }
            }
        }
    }

    def __init__(self, F, G, H, L, M, N, equivalent_stress):

        self.F = F
        self.G = G
        self.H = H
        self.L = L
        self.M = M
        self.N = N
        self.equivalent_stress = equivalent_stress

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        F, G, H = kwargs['F'], kwargs['G'], kwargs['H']
        L, M, N = kwargs['L'], kwargs['M'], kwargs['N']

        diff_sq_sum = (
            F * (stress_states[:, 1, 1] - stress_states[:, 2, 2])**2 +
            G * (stress_states[:, 2, 2] - stress_states[:, 0, 0])**2 +
            H * (stress_states[:, 0, 0] - stress_states[:, 1, 1])**2 +
            L * 2 * stress_states[:, 1, 2]**2 +
            M * 2 * stress_states[:, 2, 0]**2 +
            N * 2 * stress_states[:, 0, 1]**2
        )

        value = ((diff_sq_sum ** (1 / 2)) / kwargs['equivalent_stress']) - 1

        return value

    @property
    def lankford(self):
        'Get the Lankford coefficient (R-value)'
        return self.H / self.G


class Dummy2DYieldFunction(YieldFunction):
    'Dummy 2D yield function that resembles a cylinder about the z-axis.'

    PARAMETERS = [
        'radius',
    ]

    def __init__(self, radius):

        self.radius = radius

    @staticmethod
    @yield_function_fitter
    def residual(fitting_params, stress_states, fitting_param_names, **kwargs):

        princ = np.sort(np.linalg.eigvals(stress_states), axis=-1)[:, ::-1]

        # r = 10.4:
        # value = (np.sqrt(np.abs(princ[:, 0]**2 + princ[:, 1]**2)) / kwargs['radius']) - 1

        # r = 10:
        value = np.sqrt(np.abs(princ[:, 0]**2 + princ[:, 1]**2)) - kwargs['radius']

        return value
