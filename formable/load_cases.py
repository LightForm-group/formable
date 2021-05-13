"""`formable.load_cases.py`

Functions that generate load cases for use in simulations.

"""

import numpy as np
from vecmaths.rotation import get_random_rotation_matrix, axang2rotmat


def check_load_case(load_case):
    """Sanity checks on a load case. This should be a unit test..."""

    if load_case['def_grad_aim'] is not None:
        dg_arr = load_case['def_grad_aim']
    elif load_case['def_grad_rate'] is not None:
        dg_arr = load_case['def_grad_rate']

    if not np.alltrue(np.logical_xor(dg_arr.mask, load_case['stress'].mask)):
        raise RuntimeError(f'Stress and strain tensor masks should be element-wise '
                           f'mutually exclusive, but they are not.')

    return load_case


def get_load_case_uniaxial(total_time, num_increments, direction, target_strain_rate=None,
                           target_strain=None, rotation=None, dump_frequency=1):
    """Get a uniaxial load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    direction : str
        Either a single character, "x", "y" or "z", representing the loading direction.
    target_strain_rate : float
        Target strain rate to apply along the loading direction.
    target_strain : float
        Target strain to achieve along the loading direction.
    rotation : dict, optional
        Dict to specify a rotation of the loading direction. With keys:
            axis : ndarray of shape (3) or list of length 3
                Axis of rotation.
            angle_deg : float
                Angle of rotation about `axis` in degrees.
    dump_frequency : int, optional
        By default, 1, meaning results are written out every increment.

    Returns
    -------
    dict
        Dict representing the load case, with keys:
            direction : str
                Passed through from input argument.
            rotation : dict
                Passed through from input argument.            
            total_time : float or int
                Passed through from input argument.
            num_increments : int
                Passed through from input argument.
            rotation_matrix : ndarray of shape (3, 3), optional
                If `rotation` was specified, this will be the matrix representation of
                `rotation`.
            def_grad_rate : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient rate tensor. Not None if target_strain_rate is 
                specified. Masked values correspond to unmasked values in `stress`.
            def_grad_aim : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient aim tensor. Not None if target_strain is specified.
                Masked values correspond to unmasked values in `stress`.
            stress : numpy.ma.core.MaskedArray of shape (3, 3)
                Stress tensor. Masked values correspond to unmasked values in
                `def_grad_rate` or `def_grad_aim`.
            dump_frequency : int, optional
                Passed through from input argument.

    """

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    if rotation:
        rot_mat = axang2rotmat(
            np.array(rotation['axis']),
            rotation['angle_deg'],
            degrees=True
        )
    else:
        rot_mat = None

    if target_strain_rate is not None:
        def_grad_val = target_strain_rate
    else:
        def_grad_val = target_strain

    dir_idx = ['x', 'y', 'z']
    try:
        loading_dir_idx = dir_idx.index(direction)
    except ValueError:
        msg = (f'Loading direction "{direction}" not allowed. It should be one of "x", '
               f'"y" or "z".')
        raise ValueError(msg)

    dg_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.eye(3))
    stress_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.logical_not(np.eye(3)))

    dg_arr[loading_dir_idx, loading_dir_idx] = def_grad_val
    dg_arr.mask[loading_dir_idx, loading_dir_idx] = False
    stress_arr.mask[loading_dir_idx, loading_dir_idx] = True

    def_grad_aim = dg_arr if target_strain is not None else None
    def_grad_rate = dg_arr if target_strain_rate is not None else None

    load_case = {
        'direction': direction,
        'rotation': rotation,
        'total_time': total_time,
        'num_increments': num_increments,
        'rotation_matrix': rot_mat,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress_arr,
        'dump_frequency': dump_frequency,
    }

    return check_load_case(load_case)


def get_load_case_biaxial(total_time, num_increments, direction, target_strain_rate=None,
                          target_strain=None, rotation=None, dump_frequency=1):
    """Get a biaxial load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    direction : str
        String of two characters, ij, where {i,j} ∈ {"x","y","z"}, corresponding to the
        two loading directions.
    target_strain_rate : list of float of length two
        Target strain rates to apply along the two loading direction.
    target_strain : list of float of length two
        Target strains to achieve along the two loading directions.
    rotation : dict, optional
        Dict to specify a rotation of the loading direction. With keys:
            axis : ndarray of shape (3) or list of length 3
                Axis of rotation.
            angle_deg : float
                Angle of rotation about `axis` in degrees.        
    dump_frequency : int, optional
        By default, 1, meaning results are written out every increment.

    Returns
    -------
    dict
        Dict representing the load case, with keys:
            direction : str
                Passed through from input argument.
            rotation : dict
                Passed through from input argument.            
            total_time : float or int
                Passed through from input argument.
            num_increments : int
                Passed through from input argument.
            rotation_matrix : ndarray of shape (3, 3), optional
                If `rotation` was specified, this will be the matrix representation of
                `rotation`.
            def_grad_rate : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient rate tensor. Not None if target_strain_rate is 
                specified. Masked values correspond to unmasked values in `stress`.
            def_grad_aim : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient aim tensor. Not None if target_strain is specified.
                Masked values correspond to unmasked values in `stress`.
            stress : numpy.ma.core.MaskedArray of shape (3, 3)
                Stress tensor. Masked values correspond to unmasked values in
                `def_grad_rate` or `def_grad_aim`.
            dump_frequency : int, optional
                Passed through from input argument.

    """

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    if rotation:
        rot_mat = axang2rotmat(
            np.array(rotation['axis']),
            rotation['angle_deg'],
            degrees=True
        )
    else:
        rot_mat = None

    if target_strain_rate is not None:
        def_grad_val = target_strain_rate
    else:
        def_grad_val = target_strain

    dir_idx = ['x', 'y', 'z']
    load_dir_idx = []
    for load_dir in direction:
        try:
            loading_dir_idx = dir_idx.index(load_dir)
            load_dir_idx.append(loading_dir_idx)
        except ValueError:
            msg = (f'Loading direction "{load_dir}" not allowed. Both loading directions '
                   f'should be one of "x", "y" or "z".')
            raise ValueError(msg)

    zero_stress_dir = list(set(dir_idx) - set(direction))[0]
    zero_stress_dir_idx = dir_idx.index(zero_stress_dir)

    dg_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.zeros((3, 3)))
    stress_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.ones((3, 3)))

    dg_arr[load_dir_idx, load_dir_idx] = def_grad_val
    dg_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = True
    stress_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = False

    def_grad_aim = dg_arr if target_strain is not None else None
    def_grad_rate = dg_arr if target_strain_rate is not None else None

    load_case = {
        'direction': direction,
        'rotation': rotation,
        'total_time': total_time,
        'num_increments': num_increments,
        'rotation_matrix': rot_mat,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress_arr,
        'dump_frequency': dump_frequency,
    }

    return check_load_case(load_case)


def get_load_case_plane_strain(total_time, num_increments, direction,
                               target_strain_rate=None, target_strain=None, rotation=None,
                               dump_frequency=1):
    """Get a plane-strain load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    direction : str
        String of two characters, ij, where {i,j} ∈ {"x","y","z"}. The first character, i,
        corresponds to the loading direction and the second, j, corresponds to the
        zero-strain direction. Zero stress will be maintained on the remaining direction.
    target_strain_rate : float
        Target strain rate to apply along the loading direction.
    target_strain : float
        Target strain to achieve along the loading direction.
    rotation : dict, optional
        Dict to specify a rotation of the loading direction. With keys:
            axis : ndarray of shape (3) or list of length 3
                Axis of rotation.
            angle_deg : float
                Angle of rotation about `axis` in degrees.        
    dump_frequency : int, optional
        By default, 1, meaning results are written out every increment.

    Returns
    -------
    dict
        Dict representing the load case, with keys:
            direction : str
                Passed through from input argument.
            rotation : dict
                Passed through from input argument.            
            total_time : float or int
                Passed through from input argument.
            num_increments : int
                Passed through from input argument.
            rotation_matrix : ndarray of shape (3, 3), optional
                If `rotation` was specified, this will be the matrix representation of
                `rotation`.
            def_grad_rate : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient rate tensor. Not None if target_strain_rate is 
                specified. Masked values correspond to unmasked values in `stress`.
            def_grad_aim : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient aim tensor. Not None if target_strain is specified.
                Masked values correspond to unmasked values in `stress`.
            stress : numpy.ma.core.MaskedArray of shape (3, 3)
                Stress tensor. Masked values correspond to unmasked values in
                `def_grad_rate` or `def_grad_aim`.
            dump_frequency : int, optional
                Passed through from input argument.

    """

    # Validation:
    msg = 'Specify either `target_strain_rate` or `target_strain`.'
    if all([t is None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)
    if all([t is not None for t in [target_strain_rate, target_strain]]):
        raise ValueError(msg)

    if rotation:
        rot_mat = axang2rotmat(
            np.array(rotation['axis']),
            rotation['angle_deg'],
            degrees=True
        )
    else:
        rot_mat = None

    if target_strain_rate is not None:
        def_grad_val = target_strain_rate
    else:
        def_grad_val = target_strain

    dir_idx = ['x', 'y', 'z']
    loading_dir, zero_strain_dir = direction
    try:
        loading_dir_idx = dir_idx.index(loading_dir)
    except ValueError:
        msg = (f'Loading direction "{loading_dir}" not allowed. It should be one of "x", '
               f'"y" or "z".')
        raise ValueError(msg)

    if zero_strain_dir not in dir_idx:
        msg = (f'Zero-strain direction "{zero_strain_dir}" not allowed. It should be one '
               f'of "x", "y" or "z".')
        raise ValueError(msg)

    zero_stress_dir = list(set(dir_idx) - {loading_dir, zero_strain_dir})[0]
    zero_stress_dir_idx = dir_idx.index(zero_stress_dir)

    dg_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.zeros((3, 3)))
    stress_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.ones((3, 3)))

    dg_arr[loading_dir_idx, loading_dir_idx] = def_grad_val
    dg_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = True
    stress_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = False

    def_grad_aim = dg_arr if target_strain is not None else None
    def_grad_rate = dg_arr if target_strain_rate is not None else None

    load_case = {
        'direction': direction,
        'rotation': rotation,
        'total_time': total_time,
        'num_increments': num_increments,
        'rotation_matrix': rot_mat,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'stress': stress_arr,
        'dump_frequency': dump_frequency,
    }

    return check_load_case(load_case)


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
