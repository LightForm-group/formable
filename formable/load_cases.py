"""`formable.load_cases.py`

Functions that generate load cases for use in simulations.

"""

import numpy as np
from vecmaths.rotation import get_random_rotation_matrix, axang2rotmat


def get_load_case_uniaxial(total_time, num_increments, direction, target_strain_rate=None,
                           target_strain=None, rotation=None, dump_frequency=1):

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    if rotation:
        rotation = axang2rotmat(
            np.array(rotation['axis']),
            rotation['angle_deg'],
            degrees=True
        )

    dg_uniaxial_val = target_strain_rate or target_strain

    # TODO: refactor:
    if direction == 'x':
        dg_arr = np.ma.masked_array(
            [
                [dg_uniaxial_val, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            mask=np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
            ])
        )
    elif direction == 'y':
        dg_arr = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, dg_uniaxial_val, 0],
                [0, 0, 0]
            ],
            mask=np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ])
        )
    elif direction == 'z':
        dg_arr = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, dg_uniaxial_val]
            ],
            mask=np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ])
        )

    def_grad_aim = dg_arr if target_strain else None
    def_grad_rate = dg_arr if target_strain_rate else None

    load_case = {
        'total_time': total_time,
        'num_increments': num_increments,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress,
        'rotation': rotation,
        'dump_frequency': dump_frequency,
    }

    return load_case


def get_load_case_biaxial(total_time, num_increments, direction, target_strain_rate=None,
                          target_strain=None, dump_frequency=1):

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    dg_biaxial_vals = target_strain_rate or target_strain

    # TODO: refactor:
    if direction == 'xy':
        dg_arr = np.ma.masked_array(
            [
                [dg_biaxial_vals[0], 0, 0],
                [0, dg_biaxial_vals[1], 0],
                [0, 0, 0]
            ],
            mask=np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ])
        )
    elif direction == 'xz':
        dg_arr = np.ma.masked_array(
            [
                [dg_biaxial_vals[0], 0, 0],
                [0, 0, 0],
                [0, 0, dg_biaxial_vals[1]]
            ],
            mask=np.array([
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ])
        )
    elif direction == 'yz':
        dg_arr = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, dg_biaxial_vals[0], 0],
                [0, 0, dg_biaxial_vals[1]]
            ],
            mask=np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ])
        )

    def_grad_aim = dg_arr if target_strain else None
    def_grad_rate = dg_arr if target_strain_rate else None

    load_case = {
        'total_time': total_time,
        'num_increments': num_increments,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress,
        'dump_frequency': dump_frequency,
    }

    return load_case


def get_load_case_plane_strain(total_time, num_increments, direction,
                               target_strain_rate=None, target_strain=None,
                               dump_frequency=1):

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    dg_ps_val = target_strain_rate or target_strain

    # TODO: refactor:
    if direction == 'xy':
        dg_arr = np.ma.masked_array(
            [
                [dg_ps_val, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ],
            mask=np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ])
        )
    elif direction == 'zy':
        dg_arr = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, dg_ps_val]
            ],
            mask=np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ])
        )

        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ])
        )
    else:
        raise NotImplementedError()

    def_grad_aim = dg_arr if target_strain else None
    def_grad_rate = dg_arr if target_strain_rate else None

    load_case = {
        'total_time': total_time,
        'num_increments': num_increments,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress,
        'dump_frequency': dump_frequency,
    }

    return load_case


def get_load_case_random_2D(total_time, num_increments, normal_direction,
                            target_strain_rate=None, target_strain=None,
                            dump_frequency=1):

    def_grad_vals = (np.random.random(4) - 0.5)

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    dg_target_val = target_strain_rate or target_strain
    if target_strain:
        def_grad_vals *= dg_target_val
        def_grad_vals += np.eye(2).reshape(-1)
    elif target_strain_rate:
        def_grad_vals *= dg_target_val

    if normal_direction == 'x':
        # Deformation in the y-z plane

        dg_arr = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, def_grad_vals[0], def_grad_vals[1]],
                [0, def_grad_vals[2], def_grad_vals[3]],
            ],
            mask=np.array([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
            ])
        )
        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
            ])
        )

    elif normal_direction == 'y':
        # Deformation in the x-z plane

        dg_arr = np.ma.masked_array(
            [
                [def_grad_vals[0], 0, def_grad_vals[1]],
                [0, 0, 0],
                [def_grad_vals[2], 0, def_grad_vals[3]],
            ],
            mask=np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
            ])
        )
        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
            ])
        )

    elif normal_direction == 'z':
        # Deformation in the x-y plane

        dg_arr = np.ma.masked_array(
            [
                [def_grad_vals[0], def_grad_vals[1], 0],
                [def_grad_vals[2], def_grad_vals[3], 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ])
        )
        stress = np.ma.masked_array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            mask=np.array([
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
            ])
        )

    def_grad_aim = dg_arr if target_strain else None
    def_grad_rate = dg_arr if target_strain_rate else None

    load_case = {
        'total_time': total_time,
        'num_increments': num_increments,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress,
        'dump_frequency': dump_frequency,
    }

    return load_case


def get_load_case_random_3D(total_time, num_increments, target_strain, rotation=True,
                            rotation_max_angle=10, rotation_load_case=True,
                            non_random_rotation=None, dump_frequency=1):

    # Five stretch components, since it's a symmetric matrix and the trace must be zero:
    stretch_comps = (np.random.random((5,)) - 0.5) * target_strain
    stretch = np.zeros((3, 3)) * np.nan

    # Diagonal comps:
    stretch[[0, 1], [0, 1]] = stretch_comps[:2]
    stretch[2, 2] = -(stretch[0, 0] + stretch[1, 1])

    # Off-diagonal comps:
    stretch[[1, 0], [0, 1]] = stretch_comps[2]
    stretch[[2, 0], [0, 2]] = stretch_comps[3]
    stretch[[1, 2], [2, 1]] = stretch_comps[4]

    # Add the identity:
    U = stretch + np.eye(3)

    defgrad = U
    rot = None
    if rotation and non_random_rotation is None:
        rot = get_random_rotation_matrix(
            method='axis_angle',
            max_angle_deg=rotation_max_angle
        )
        if not rotation_load_case:
            defgrad = rot @ U
            rot = None

    if non_random_rotation:
        rot = axang2rotmat(
            np.array(non_random_rotation['axis']),
            non_random_rotation['angle_deg'],
            degrees=True
        )

    # Ensure defgrad has a unit determinant:
    defgrad = defgrad / (np.linalg.det(defgrad)**(1/3))

    dg_arr = np.ma.masked_array(defgrad, mask=np.zeros((3, 3), dtype=int))
    stress_arr = np.ma.masked_array(
        np.zeros((3, 3), dtype=int),
        mask=np.ones((3, 3), dtype=int)
    )

    load_case = {
        'total_time': total_time,
        'num_increments': num_increments,
        'def_grad_aim': dg_arr,
        'stress': stress_arr,
        'rotation': rot,
        'dump_frequency': dump_frequency,
    }

    return load_case
