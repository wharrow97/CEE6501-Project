import sys
import os
import numpy as np
import pandas as pd

def load_model(filename):
    with open(filename, "r" as f:
              model = json.load(f)

    nodes = {int(k): v for k, v in model["nodes"].items()}
    elements = {int(k): v for k, v in model["elements"].items()}
    supports = {int(k): v for k, v in model.get("supports", {}).items()}

    return nodes, elements, supports nodal_loads

def get_node_dof_map(nodes):
    dof_map = {}
    for i, node_id in enumerate(sorted(nodes.keys())):
        start = 6 * i + 1
        dof_map[node_id] = [start, start + 1, start + 2, start + 3, start + 4, start + 5]
    return dof_map

def direction_cosines_3d(xi, yi, zi, xj, yj, zj):
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    lx = dx / L
    my = dy / L
    nz = dz / L

    return L, lx, my, nz

def local_axes_from_element(xi, yi, zi, xj, yj, zj):
    x_vec = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
    L = np.linalg.norm(x_vec)
    ex = x_vec / L

    global_z = np.array([0.0, 0.0, 1.0])
    global_y = np.array([0.0, 1.0, 0.0])

    if np.linalg.norm(np.cross(global_z, ex)) < 1e-8:
        ref = global_y
    else:
        ref = global_z
    
    ey = np.cross(ref, ex)
    ey = ey / np.linalg.norm(ey)

    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)

    R = np.vstack((ex, ey, ez))
    return L, R

def k_local_3d_truss(E, A, L):
    k = (E * A / L) * np.array(
        [
            [1, -1],
            [-1, 1],
        ],
        dtype=float,
    )
    return k

def T_3d_truss_rotation(lx, my, nz):
    T = np.array(
        [
            [lx, my, nz, 0, 0, 0],
            [0, 0, 0, lx, my, nz],
        ],
        dtype=float,
    )
    return T

def k_local_3d_frame(E, G, A, I, J, L):
    EA_L = E * A / L
    GJ_L = G * J / L
    EI = E * I

    k = np.array(
        [
            [EA_L, 0, 0, 0, 0, 0, -EA_L, 0, 0, 0, 0, 0],
            [0, 12*EI/L**3, 0, 0, 0, 6*EI/L**2, 0, -12*EI/L**3, 0, 0, 0, 6*EI/L**2],
            [0, 0, 12*EI/L**3, 0, -6*EI/L**2, 0, 0, 0, -12*EI/L**3, 0, -6*EI/L**2, 0],
            [0, 0, 0, GJ_L, 0, 0, 0, 0, 0, -GJ_L, 0, 0],
            [0, 0, -6*EI/L**2, 0, 4*EI/L, 0, 0, 0, 6*EI/L**2, 0, 2*EI/L, 0],
            [0, 6*EI/L**2, 0, 0, 0, 4*EI/L, 0, -6*EI/L**2, 0, 0, 0, 2*EI/L],
            [-EA_L, 0, 0, 0, 0, 0, EA_L, 0, 0, 0, 0, 0],
            [0, -12*EI/L**3, 0, 0, 0, -6*EI/L**2, 0, 12*EI/L**3, 0, 0, 0, -6*EI/L**2],
            [0, 0, -12*EI/L**3, 0, 6*EI/L**2, 0, 0, 0, 12*EI/L**3, 0, 6*EI/L**2, 0],
            [0, 0, 0, -GJ_L, 0, 0, 0, 0, 0, GJ_L, 0, 0],
            [0, 0, -6*EI/L**2, 0, 2*EI/L, 0, 0, 0, 6*EI/L**2, 0, 4*EI/L, 0],
            [0, 6*EI/L**2, 0, 0, 0, 2*EI/L, 0, -6*EI/L**2, 0, 0, 0, 4*EI/L],
        ],
        dtype=float,
    )
    return k

def T_3d_frame_rotation(R):
    T = np.zeros((12, 12), dtype=float)
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    T[6:9, 6:9] = R
    T[9:12, 9:12] = R
    return T


def build_global_load_vector(nodes, dof_map, nodal_loads):
    ndof = 6 * len(nodes)
    f = np.zeros(ndof)

    for node_id, load_vals in nodal_loads.items():
        map_i = dof_map[node_id]
        for a in range(6):
            f[map_i[a] - 1] += load_vals[a]

    return f


def build_restrained_dofs(dof_map, supports):
    restrained = []

    for node_id, bc in supports.items():
        node_dofs = dof_map[node_id]
        for i in range(6):
            if bc[i] == 1:
                restrained.append(node_dofs[i])

    return restrained

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

def functions_main(input_file):
    nodes, elements, supports, nodal_loads = load_model(input_file)

    dof_map = get_node_dof_map(nodes)
    ndof = 6 * len(nodes)

    k_list = []
    T_list = []
    Qf_list = []
    map_list = []
    element_results = []

    for elem_id, elem in elements.items():
        elem_type = elem["type"].lower()
        ni, nj = elem["nodes"]

        xi, yi, zi = nodes[ni]
        xj, yj, zj = nodes[nj]

        if elem_type == "3d_truss":
            E = elem["E"]
            A = elem["A"]

            L, lx, my, nz = direction_cosines_3d(xi, yi, zi, xj, yj, zj)
            k_local_small = k_local_3d_truss(E, A, L)
            T_small = T_3d_truss_rotation(lx, my, nz)

            k_local = k_local_small
            T = T_small
            Qf = np.zeros(2)

            dofs_i = dof_map[ni][0:3]
            dofs_j = dof_map[nj][0:3]
            elem_map = dofs_i + dofs_j

            k_list.append(k_local)
            T_list.append(T)
            Qf_list.append(Qf)
            map_list.append(elem_map)

            element_results.append(
                {
                    "element_id": elem_id,
                    "type": "3d_truss",
                    "nodes": [ni, nj],
                    "L": L,
                    "E": E,
                    "A": A,
                    "map": elem_map,
                    "T": T,
                    "k_local": k_local,
                }
            )

        elif elem_type == "3d_frame":
            E = elem["E"]
            G = elem["G"]
            A = elem["A"]
            I = elem["I"]
            J = elem["J"]

            L, R = local_axes_from_element(xi, yi, zi, xj, yj, zj)
            k_local = k_local_3d_frame(E, G, A, I, J, L)
            T = T_3d_frame_rotation(R)
            Qf = np.zeros(12)

            elem_map = dof_map[ni] + dof_map[nj]

            k_list.append(k_local)
            T_list.append(T)
            Qf_list.append(Qf)
            map_list.append(elem_map)

            element_results.append(
                {
                    "element_id": elem_id,
                    "type": "3d_frame",
                    "nodes": [ni, nj],
                    "L": L,
                    "E": E,
                    "G": G,
                    "A": A,
                    "I": I,
                    "J": J,
                    "map": elem_map,
                    "T": T,
                    "k_local": k_local,
                }
            )

        else:
            raise ValueError(f"Unknown element type: {elem_type}")

    K_global, F_fef_global = assemble_global_stiffness_and_fef(
        ndof,
        k_list,
        T_list,
        Qf_list,
        map_list,
    )

    f_global = build_global_load_vector(nodes, dof_map, nodal_loads)
    dof_restrained_1based = build_restrained_dofs(dof_map, supports)

    u_known = np.zeros(ndof)

    (
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
    ) = partition_system(
        K_global,
        f_global,
        u_known,
        F_fef_global,
        dof_restrained_1based,
    )

    rhs = f_f - f_fef_f - K_fr @ u_r
    u_f = np.linalg.solve(K_ff, rhs)

    F_r = K_rf @ u_f + K_rr @ u_r + f_fef_r

    u_global = assemble_global_displacements(u_f, u_r, free_dofs, restrained_dofs)
    f_global_complete = assemble_global_forces(f_f, F_r, free_dofs, restrained_dofs)

    for elem in element_results:
        idx = [d - 1 for d in elem["map"]]
        u_elem_global = u_global[idx]
        u_elem_local = elem["T"] @ u_elem_global
        q_elem_local = elem["k_local"] @ u_elem_local

        elem["u_global"] = u_elem_global
        elem["u_local"] = u_elem_local
        elem["q_local"] = q_elem_local

        if elem["type"] == "3d_truss":
            elem["axial_force"] = q_elem_local[1]
        else:
            elem["axial_force"] = q_elem_local[6]

    results = {
        "nodes": nodes,
        "elements": elements,
        "supports": supports,
        "nodal_loads": nodal_loads,
        "dof_map": dof_map,
        "ndof": ndof,
        "K_global": K_global,
        "F_fef_global": F_fef_global,
        "u_global": u_global,
        "f_global_complete": f_global_complete,
        "dof_restrained_1based": dof_restrained_1based,
        "element_results": element_results,
    }

    return results