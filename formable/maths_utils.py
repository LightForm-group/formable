"""`formable.maths_utils.py`"""

import numpy as np
from vecmaths import vectors


def get_principal_values(tensors):
    """Get the principle values of a stress/strain tensor.

    Parameters
    ----------
    tensors : ndarray of inner shape (3, 3)

    Returns
    -------
    ndarray of inner shape (3,)
        Principal values of each tensor, ordered from largest to smallest.

    """

    return np.sort(np.linalg.eigvals(tensors), axis=-1)[..., ::-1]


def get_plane_coords(a, b, c, side_length, resolution, up=None):

    vec_a = np.array([a, b, c]).astype(float)

    if up is not None:
        # Check `up` is perpendicular to `vec_a`:
        up = np.array(up)
        if not np.isclose(np.dot(vec_a, up), 0):
            raise ValueError('`up` vector is not perpendicular to normal vector.')
        vec_c = up
    else:
        vec_c = vectors.perpendicular(vec_a)

    vec_b = np.cross(vec_a, vec_c).astype(float)
    vec_c = vec_c.astype(float)

    # If not right-handed, flip first 2D basis vector:
    vol = np.dot(
        np.cross(
            vec_a,
            vec_b,
        ),
        vec_c
    )
    if vol < 0:
        vec_b *= -1

    vec_b = vectors.normalise(vec_b) * side_length / resolution
    vec_c = vectors.normalise(vec_c) * side_length / resolution

    basis = np.hstack([vec_b[:, None], vec_c[:, None]])
    basis_unit = vectors.normalise(basis, axis=0)

    steps = np.arange(resolution + 1) - (resolution / 2)
    x, y = np.meshgrid(*[steps] * 2)

    xy = np.array([x.flatten(), y.flatten()])

    xy_corners = np.array([
        xy[:,  0],
        xy[:,  resolution],
        xy[:, -1],
        xy[:,  -(resolution + 1)],
    ]).T

    grid_coords_2D = xy * np.linalg.norm(basis, axis=0)[:, None]
    grid_coords = basis @ xy
    grid_corners = basis @ xy_corners

    out = {
        'grid_coords_2D': grid_coords_2D,
        'grid_coords': grid_coords,
        'grid_corners': grid_corners,
        'basis_unit': basis_unit,
    }

    return out


def to_symmetric_voigt_notation(tensors):
    """Convert stacks of tensors into symmetric Voigt notation.

    Parameters
    ----------
    tensors : ndarray with inner shape (3, 3)

    Returns
    -------
    sym_tensors : ndarray with inner shape (6,)


    Notes
    -----
    For each (3, 3) input tensor, the average of the opposite diagonals are used
    for the final three components of the ouput tensor.

    For an input tensor:
    [[a, b, c]
     [d, e, f]
     [g, h, i]]

    The output tensor is:
    [a, e, i, (h+f)/2, (g+c)/2, (d+b)/2]

    """

    sym_tensors = np.zeros(tensors.shape[:-2] + (6,)) * np.nan

    sym_tensors[..., [0, 1, 2]] = tensors[..., [0, 1, 2, ], [0, 1, 2]]
    sym_tensors[..., 3] = (tensors[..., 1, 2] + tensors[..., 2, 1]) / 2
    sym_tensors[..., 4] = (tensors[..., 0, 2] + tensors[..., 2, 0]) / 2
    sym_tensors[..., 5] = (tensors[..., 0, 1] + tensors[..., 1, 0]) / 2

    return sym_tensors


def from_voigt_notation(voigt_tensors):
    """Convert stacks of Voigt-notation tensors into 3x3 tensors.

    Parameters
    ----------
    voigt_tensors : ndarray with inner shape (6,)

    Returns
    -------
    tensors : ndarray with inner shape (3, 3)

    Notes
    -----
    For an input tensor: [a, b, c, d, e, f], the output tensor is:

    [[a, f, e]
     [f, b, d]
     [e, d, c]]

    """

    tensors = np.zeros(voigt_tensors.shape[:-1] + (3, 3)) * np.nan

    tensors[..., [0, 1, 2], [0, 1, 2]] = voigt_tensors[..., [0, 1, 2]]
    tensors[..., [0, 1], [1, 0]] = voigt_tensors[..., [5, 5]]
    tensors[..., [0, 2], [2, 0]] = voigt_tensors[..., [4, 4]]
    tensors[..., [1, 2], [2, 1]] = voigt_tensors[..., [3, 3]]

    return tensors


def get_plane_stress_principle_stresses(stresses):
    """
    Parameters
    ----------
    stresses : ndarray of shape (3, N)
        Array of column vectors where each column vector is a single stress state
        that specifies: s_11, s_22, and s_12.

    Returns
    -------
    princ_stresses : ndarray (2, N)
        Array of column vectors where each column vector is a single principle
        stress state that specifies: s1, s2.

    """

    A = stresses[0] + stresses[1]
    B = np.sqrt((stresses[0] - stresses[1])**2 + (4 * stresses[2]**2))

    s1 = 0.5 * (A + B)
    s2 = 0.5 * (A - B)

    princ_stresses = np.vstack([s1, s2])

    return princ_stresses


def get_deviatoric_stress(stress, voigt=False):
    """Get stress deviators for stacks of stress matrices.

    Parameters
    ----------
    stress : ndarray
        If `voigt` is True, inner shape must be (6,), otherwise inner shape
        must be (3, 3).
    voigt : bool, optional
        If True, expect input (and return output) stresses in Voigt notation.

    Returns
    -------
    stress_dev : ndarray
        Deviatoric stresses. If `voigt` is True, inner shape will be (6,),
        otherwise inner shape will be (3, 3).

    """

    if voigt:
        hydro = np.sum(stress[..., 0:3], axis=-1) / 3
        hydro_rep = np.repeat(hydro[..., None], 3, axis=-1)
        zeros = np.zeros_like(hydro_rep)
        hydro_mat = np.concatenate([hydro_rep, zeros], axis=-1)

    else:
        trace = np.trace(stress, axis1=-2, axis2=-1)
        hydro = (np.trace(stress, axis1=-2, axis2=-1)) / 3
        hydro_rep = np.repeat(hydro[..., None], 3, axis=-1)
        hydro_mat = np.zeros(stress.shape)
        hydro_mat[..., [0, 1, 2], [0, 1, 2]] = hydro_rep

    stress_dev = stress - hydro_mat

    return stress_dev


def line(x, m, c):
    'Linear model.'
    return (m * x) + c
