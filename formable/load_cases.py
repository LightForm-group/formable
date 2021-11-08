"""`formable.load_cases.py`

Functions that generate load cases for use in simulations.

"""

import copy

import numpy as np
from numpy.linalg.linalg import norm
from vecmaths.rotation import get_random_rotation_matrix, axang2rotmat


def check_load_case(load_case):
    """Sanity checks on a load case. This should be a unit test..."""

    if load_case['def_grad_aim'] is not None:
        dg_arr = load_case['def_grad_aim']
    elif load_case['def_grad_rate'] is not None:
        dg_arr = load_case['def_grad_rate']
    elif load_case['vel_grad'] is not None:
        dg_arr = load_case['vel_grad']

    if load_case['stress'] is not None:
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
                               dump_frequency=1, strain_rate_mode=None):
    """Get a plane-strain load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    direction : str
        String of two characters, ij, where {i,j} ∈ {"x","y","z"}. The first character, i,
        corresponds to the loading direction and the second, j, corresponds to the
        zero-strain direction. Zero stress will be specified on the remaining direction.
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
    strain_rate_mode : str, optional
        One of "F_rate", "L", "L_approx". If not specified, default is "F_rate". Use
        "L_approx" for specifying non-mixed boundary conditions.

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
                specified and `strain_rate_mode` is None or "F_rate". Masked values
                correspond to unmasked values in `stress`.
            def_grad_aim : numpy.ma.core.MaskedArray of shape (3, 3), optional
                Deformation gradient aim tensor. Not None if target_strain is specified.
                Masked values correspond to unmasked values in `stress`.
            vel_grad : (ndarray or numpy.ma.core.MaskedArray) of shape (3, 3), optional
                Velocity gradient aim tensor. Not None if `strain_rate_mode` is one of "L"
                (will be a masked array) or "L_approx" (will be an ordinary array).
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

    if strain_rate_mode is None:
        strain_rate_mode = 'F_rate'
    if strain_rate_mode not in ['F_rate', 'L', 'L_approx']:
        msg = 'Strain rate mode must be `F_rate`, `L` or `L_approx`.'
        raise ValueError(msg)
    if strain_rate_mode in ['L', 'L_approx'] and target_strain_rate is None:
        msg = (f'`target_strain_rate` must be specified for `strain_rate_mode`'
               f'`{strain_rate_mode}`')
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

    if strain_rate_mode == 'L':
        # When using L with mixed BCs, each row must be either L or P:
        dg_arr.mask[zero_stress_dir_idx] = True
        stress_arr.mask[zero_stress_dir_idx] = False

    elif strain_rate_mode == 'L_approx':
        dg_arr = dg_arr.data  # No need for a masked array
        # Without mixed BCs, we can get volume conservation with Trace(L) = 0:
        dg_arr[zero_stress_dir_idx, zero_stress_dir_idx] = -def_grad_val
        stress_arr = None

    elif strain_rate_mode == 'F_rate':
        dg_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = True
        stress_arr.mask[zero_stress_dir_idx, zero_stress_dir_idx] = False

    if strain_rate_mode in ['L', 'L_approx']:
        def_grad_aim = None
        def_grad_rate = None
        vel_grad = dg_arr
    else:
        def_grad_aim = dg_arr if target_strain is not None else None
        def_grad_rate = dg_arr if target_strain_rate is not None else None
        vel_grad = None

    load_case = {
        'direction': direction,
        'rotation': rotation,
        'total_time': total_time,
        'num_increments': num_increments,
        'rotation_matrix': rot_mat,
        'def_grad_rate': def_grad_rate,
        'def_grad_aim': def_grad_aim,
        'vel_grad': vel_grad,
        'stress': stress_arr,
        'dump_frequency': dump_frequency,
    }

    return check_load_case(load_case)


def get_load_case_planar_2D(total_time, num_increments, normal_direction,
                            target_strain_rate=None, target_strain=None, rotation=None,
                            dump_frequency=1):
    """Get a planar 2D load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    normal_direction : str
        A single character, "x", "y" or "z", representing the loading plane normal
        direction.
    target_strain_rate : (nested) list of float or ndarray of shape (2, 2)
        Target deformation gradient rate components. Either a 2D array, nested list, or a
        flat list. If passed as a flat list, the first and fourth elements correspond to
        the normal components of the deformation gradient rate tensor. The second element
        corresponds to the first-row, second-column (shear) component and the third
        element corresponds to the second-row, first-column (shear) component.
    target_strain : (nested) list of float or ndarray of shape (2, 2)
        Target deformation gradient components. Either a 2D array, nested list, or a
        flat list. If passed as a flat list, the first and fourth elements correspond to
        the normal components of the deformation gradient tensor. The second element
        corresponds to the first-row, second-column (shear) component and the third
        element corresponds to the second-row, first-column (shear) component.
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
            normal_direction : str
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
        def_grad_vals = target_strain_rate
    else:
        def_grad_vals = target_strain

    # Flatten list/array:
    if isinstance(def_grad_vals, list):
        if isinstance(def_grad_vals[0], list):
            def_grad_vals = [j for i in def_grad_vals for j in i]
    elif isinstance(def_grad_vals, np.ndarray):
        def_grad_vals = def_grad_vals.flatten()

    dir_idx = ['x', 'y', 'z']
    try:
        normal_dir_idx = dir_idx.index(normal_direction)
    except ValueError:
        msg = (f'Normal direction "{normal_direction}" not allowed. It should be one of '
               f'"x", "y" or "z".')
        raise ValueError(msg)

    loading_col_idx = list({0, 1, 2} - {normal_dir_idx})
    dg_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.zeros((3, 3)))
    stress_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.zeros((3, 3)))

    dg_row_idx = [
        loading_col_idx[0],
        loading_col_idx[0],
        loading_col_idx[1],
        loading_col_idx[1],
    ]
    dg_col_idx = [
        loading_col_idx[0],
        loading_col_idx[1],
        loading_col_idx[0],
        loading_col_idx[1],
    ]
    dg_arr[dg_row_idx, dg_col_idx] = def_grad_vals
    dg_arr.mask[:, normal_dir_idx] = True
    stress_arr.mask[:, loading_col_idx] = True

    def_grad_aim = dg_arr if target_strain is not None else None
    def_grad_rate = dg_arr if target_strain_rate is not None else None

    load_case = {
        'normal_direction': normal_direction,
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
                            target_strain_rate=None, target_strain=None, rotation=None,
                            dump_frequency=1):
    """Get a random 2D planar load case.

    Parameters
    ----------
    total_time : float or int
    num_increments : int
    normal_direction : str
        A single character, "x", "y" or "z", representing the loading plane normal
        direction.
    target_strain_rate : float
        Maximum target deformation gradient rate component. Components will be sampled
        randomly in the inteval [-target_strain_rate, +target_strain_rate).
    target_strain : float
        Maximum target deformation gradient component. Components will be sampled
        randomly in the inteval [-target_strain, +target_strain).
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
            normal_direction : str
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

    def_grad_vals = (np.random.random(4) - 0.5) * 2
    if target_strain_rate is not None:
        target_strain_rate *= def_grad_vals
    else:
        target_strain *= def_grad_vals
        target_strain += np.eye(2).reshape(-1)

    load_case = get_load_case_planar_2D(
        total_time=total_time,
        num_increments=num_increments,
        normal_direction=normal_direction,
        target_strain=target_strain,
        target_strain_rate=target_strain_rate,
        rotation=rotation,
        dump_frequency=dump_frequency,
    )
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


def get_load_case_uniaxial_cyclic(max_stress, min_stress, cycle_frequency,
                                  num_increments_per_cycle, num_cycles, direction,
                                  waveform, dump_frequency=1):

    dir_idx = ['x', 'y', 'z']
    try:
        loading_dir_idx = dir_idx.index(direction)
    except ValueError:
        msg = (f'Loading direction "{direction}" not allowed. It should be one of "x", '
               f'"y" or "z".')
        raise ValueError(msg)

    cycle_time = 1 / cycle_frequency

    if waveform.lower() == 'sine':

        sig_mean = (max_stress + min_stress) / 2
        sig_diff = max_stress - min_stress

        A = 2 * np.pi / cycle_time
        time = np.linspace(0, 2 * np.pi, num=num_increments_per_cycle, endpoint=True) / A
        sig = (sig_diff / 2) * np.sin(A * time) + sig_mean

        time_per_inc = cycle_time / num_increments_per_cycle

        stress_mask = np.ones((sig.size, 3, 3))
        stress_mask[:, [0, 1, 2], [0, 1, 2]] = 0
        stress_arr = np.ma.masked_array(
            data=np.zeros((sig.size, 3, 3)),
            mask=stress_mask,
        )
        stress_arr[:, loading_dir_idx, loading_dir_idx] = sig
        
        dg_arr = np.ma.masked_array(np.zeros((3, 3)), mask=np.eye(3))
        
        cycle = []
        for time_idx, time_i in enumerate(time):
            cycle.append({
                'num_increments': 1,
                'total_time': time_per_inc,
                'stress': stress_arr[time_idx],
                'def_grad_aim': dg_arr,
                'dump_frequency': dump_frequency,
            })
            
        out = []
        for cycle_idx in range(num_cycles):
            cycle_i = copy.deepcopy(cycle)
            if cycle_idx != num_cycles - 1:
                 # intermediate cycle; remove repeated increment:
                 cycle_i = cycle_i[:-1]
            out.extend(cycle_i)

    else:
        raise NotImplementedError('Only waveform "sine" is currently allowed.')
    
    return out
