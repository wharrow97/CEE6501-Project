import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def assemble_global_stiffness_and_fef(
    ndof,
    k_list,
    T_list,
    Qf_list,
    map_list,
):
    """
    Assemble global stiffness matrix and global fixed-end force vector.

    Automatically handles 6-DOF (frame) and 4-DOF (truss/beam) elements.
    Parameters
    ----------
    ndof : int
        Total number of global degrees of freedom.

    k_list : list of ndarray
        List of local element stiffness matrices.
        Each matrix may be 6x6 (frame) or 4x4 (truss/beam).

    T_list : list of ndarray
        List of element transformation matrices corresponding
        to each k_local. Must be compatible in size.

    Qf_list : list of ndarray
        List of local fixed-end force vectors for each element.
        Size must match the element DOF count.

    map_list : list of array-like
        List of element DOF maps (1-based indexing).
        Each map defines where the element DOFs connect
        into the global DOF numbering.

    Returns
    -------
    K_global : ndarray (ndof x ndof)
        Assembled global stiffness matrix.

    F_fef_global : ndarray (ndof,)
        Assembled global fixed-end force vector.

    Notes
    -----
    - DOF maps are assumed to use 1-based indexing.
    - Internally converted to 0-based indexing for Python.
    - Assembly is dense; for large systems a sparse format
      should be used instead.
    """

    K_global = np.zeros((ndof, ndof), dtype=float)
    F_fef_global = np.zeros(ndof, dtype=float)

    nelem = len(k_list)

    for i in range(nelem):

        k_local = k_list[i]
        T = T_list[i]
        Qf_local = Qf_list[i]
        dof_map = map_list[i]  # 1-based indexing

        # Determine element DOF count automatically
        edof = k_local.shape[0]

        # Transform to global
        K = T.T @ k_local @ T
        F_fef = T.T @ Qf_local

        # Scatter-add
        for a in range(edof):
            A = dof_map[a] - 1  # convert to 0-based

            F_fef_global[A] += F_fef[a]

            for b in range(edof):
                B = dof_map[b] - 1
                K_global[A, B] += K[a, b]

    return K_global, F_fef_global


def partition_system(K, f, u, f_fef, dof_restrained_1based):
    ndof = K.shape[0]

    # Convert restrained DOFs to 0-based
    restrained_dofs = sorted(int(d) - 1 for d in dof_restrained_1based)

    # Free DOFs
    free_dofs = [i for i in range(ndof) if i not in restrained_dofs]

    # Partition stiffness matrix
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fr = K[np.ix_(free_dofs, restrained_dofs)]
    K_rf = K[np.ix_(restrained_dofs, free_dofs)]
    K_rr = K[np.ix_(restrained_dofs, restrained_dofs)]

    # Partition force vector
    f_f = f[free_dofs]
    f_r = f[restrained_dofs]

    # Partition displaced vector
    u_r = u[restrained_dofs]

    # Partition fixed-end forces
    f_fef_f = f_fef[free_dofs]
    f_fef_r = f_fef[restrained_dofs]

    return (
        K_ff,
        K_fr,
        K_rf,
        K_rr,
        f_f,
        f_r,
        u_r,
        f_fef_f,
        f_fef_r,
        free_dofs,
        restrained_dofs,
    )


def assemble_global_displacements(u_f, u_r, free_dofs, restrained_dofs):
    """
    Assemble the full global displacement vector u from partitioned results.
    """
    ndof_total = len(free_dofs) + len(restrained_dofs)
    u_global = np.zeros(ndof_total)

    if u_r is None:
        u_r = np.zeros(len(restrained_dofs))

    u_global[free_dofs] = u_f
    u_global[restrained_dofs] = u_r

    return u_global


def assemble_global_forces(f_f, F_r, free_dofs, restrained_dofs):
    """
    Assemble the full global force vector f from applied loads and reactions.
    """
    ndof_total = len(free_dofs) + len(restrained_dofs)
    f_global = np.zeros(ndof_total)

    f_global[free_dofs] = f_f
    f_global[restrained_dofs] = F_r

    return f_global


def fef_local_2d_frame_point_midspan_moment_release(P, L, release=None):
    """
    Local fixed-end-force vector
    for a transverse point load P at midspan.

    DOF order:
        [u1, v1, th1, u2, v2, th2]
    """
    if release is None:
        return np.array(
            [0, P / 2, P * L / 8, 0, P / 2, -P * L / 8], dtype=float
        )

    elif release == "MT1":
        return np.array(
            [0, 3 * P / 8, 0, 0, 5 * P / 8, -P * L / 8], dtype=float
        )

    elif release == "MT2":
        return np.array(
            [0, 5 * P / 8, P * L / 8, 0, 3 * P / 8, 0], dtype=float
        )

    elif release in ("MT1_MT2", "both"):
        return np.array([0, P / 2, 0, 0, P / 2, 0], dtype=float)

    else:
        raise ValueError(
            "release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'"
        )


def fef_local_2d_frame_udl_moment_release(w, L, release=None):
    """
    Local fixed-end-force vector
    for a full-span transverse UDL on a 2D frame element.

    DOF order:
        [u1, v1, th1, u2, v2, th2]

    Sign convention:
        This returns the standard local element fixed-end-force vector
        consistent with:
            fixed: [0, wL/2, wL^2/12, 0, wL/2, -wL^2/12]
    """
    if release is None:
        return np.array(
            [0, w * L / 2, w * L**2 / 12, 0, w * L / 2, -w * L**2 / 12],
            dtype=float,
        )

    elif release == "MT1":
        return np.array(
            [0, 3 * w * L / 8, 0, 0, 5 * w * L / 8, -w * L**2 / 8], dtype=float
        )

    elif release == "MT2":
        return np.array(
            [0, 5 * w * L / 8, w * L**2 / 8, 0, 3 * w * L / 8, 0], dtype=float
        )

    elif release in ("MT1_MT2", "both"):
        return np.array([0, w * L / 2, 0, 0, w * L / 2, 0], dtype=float)

    else:
        raise ValueError(
            "release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'"
        )


def k_local_2d_frame_moment_release(E, A, I_z, L, release=None):
    """
    Local stiffness matrix for a 2D planar frame element.

    DOF order:
        [u1, v1, th1, u2, v2, th2]

    Parameters
    ----------
    E : float
        Young's modulus
    A : float
        Cross-sectional area
    I_z : float
        Second moment of area
    L : float
        Element length
    release : str or None
        None  : fully fixed frame element
        "MT1" : moment release at end 1
        "MT2" : moment release at end 2
        "MT1_MT2" or "both" : moment releases at both ends

    Returns
    -------
    k : (6,6) ndarray
        Local element stiffness matrix
    """
    EA_L = E * A / L
    EI = E * I_z

    # axial part (always unchanged)
    k_axial = np.array(
        [
            [EA_L, 0.0, 0.0, -EA_L, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-EA_L, 0.0, 0.0, EA_L, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    if release is None:
        # standard 2D frame bending part
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [
                    0,
                    12 * EI / L**3,
                    6 * EI / L**2,
                    0,
                    -12 * EI / L**3,
                    6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 4 * EI / L, 0, -6 * EI / L**2, 2 * EI / L],
                [0, 0, 0, 0, 0, 0],
                [
                    0,
                    -12 * EI / L**3,
                    -6 * EI / L**2,
                    0,
                    12 * EI / L**3,
                    -6 * EI / L**2,
                ],
                [0, 6 * EI / L**2, 2 * EI / L, 0, -6 * EI / L**2, 4 * EI / L],
            ],
            dtype=float,
        )

    elif release == "MT1":
        # end 1 moment released
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3 * EI / L**3, 0, 0, -3 * EI / L**3, 3 * EI / L**2],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -3 * EI / L**3, 0, 0, 3 * EI / L**3, -3 * EI / L**2],
                [0, 3 * EI / L**2, 0, 0, -3 * EI / L**2, 3 * EI / L],
            ],
            dtype=float,
        )

    elif release == "MT2":
        # end 2 moment released
        k_bend = np.array(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 3 * EI / L**3, 3 * EI / L**2, 0, -3 * EI / L**3, 0],
                [0, 3 * EI / L**2, 3 * EI / L, 0, -3 * EI / L**2, 0],
                [0, 0, 0, 0, 0, 0],
                [0, -3 * EI / L**3, -3 * EI / L**2, 0, 3 * EI / L**3, 0],
                [0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

    elif release in ("MT1_MT2", "both"):
        # both ends moment released -> truss-like bending behavior
        k_bend = np.zeros((6, 6), dtype=float)

    else:
        raise ValueError(
            "release must be None, 'MT1', 'MT2', or 'MT1_MT2'/'both'"
        )

    return k_axial + k_bend


def frame_transformation_2d(theta_deg):
    theta = np.deg2rad(theta_deg)
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array(
        [
            [c, s, 0, 0, 0, 0],
            [-s, c, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, c, s, 0],
            [0, 0, 0, -s, c, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=float,
    )

    return T
