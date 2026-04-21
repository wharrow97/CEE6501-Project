import os
import json
import numpy as np


def load_model(filename):
    with open(filename, "r") as f:
        model = json.load(f)

    nodes = {int(k): v for k, v in model["nodes"].items()}
    elements = {int(k): v for k, v in model["elements"].items()}
    supports = {int(k): v for k, v in model.get("supports", {}).items()}
    nodal_loads = {int(k): v for k, v in model.get("nodal_loads", {}).items()}

    return nodes, elements, supports, nodal_loads


def get_node_dof_map(nodes):
    dof_map = {}
    sorted_nodes = sorted(nodes.keys())

    for i, node_id in enumerate(sorted_nodes):
        start = 6 * i + 1
        dof_map[node_id] = [start, start + 1, start + 2, start + 3, start + 4, start + 5]

    return dof_map


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


def k_local_3d_frame(E, G, A, I, J, L):
    EA_L = E * A / L
    GJ_L = G * J / L
    EI = E * I

    k = np.array(
        [
            [EA_L, 0, 0, 0, 0, 0, -EA_L, 0, 0, 0, 0, 0],
            [0, 12 * EI / L**3, 0, 0, 0, 6 * EI / L**2, 0, -12 * EI / L**3, 0, 0, 0, 6 * EI / L**2],
            [0, 0, 12 * EI / L**3, 0, -6 * EI / L**2, 0, 0, 0, -12 * EI / L**3, 0, -6 * EI / L**2, 0],
            [0, 0, 0, GJ_L, 0, 0, 0, 0, 0, -GJ_L, 0, 0],
            [0, 0, -6 * EI / L**2, 0, 4 * EI / L, 0, 0, 0, 6 * EI / L**2, 0, 2 * EI / L, 0],
            [0, 6 * EI / L**2, 0, 0, 0, 4 * EI / L, 0, -6 * EI / L**2, 0, 0, 0, 2 * EI / L],
            [-EA_L, 0, 0, 0, 0, 0, EA_L, 0, 0, 0, 0, 0],
            [0, -12 * EI / L**3, 0, 0, 0, -6 * EI / L**2, 0, 12 * EI / L**3, 0, 0, 0, -6 * EI / L**2],
            [0, 0, -12 * EI / L**3, 0, 6 * EI / L**2, 0, 0, 0, 12 * EI / L**3, 0, 6 * EI / L**2, 0],
            [0, 0, 0, -GJ_L, 0, 0, 0, 0, 0, GJ_L, 0, 0],
            [0, 0, -6 * EI / L**2, 0, 2 * EI / L, 0, 0, 0, 6 * EI / L**2, 0, 4 * EI / L, 0],
            [0, 6 * EI / L**2, 0, 0, 0, 2 * EI / L, 0, -6 * EI / L**2, 0, 0, 0, 4 * EI / L],
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


def k_global_3d_truss(E, A, xi, yi, zi, xj, yj, zj):
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi

    L = np.sqrt(dx**2 + dy**2 + dz**2)

    l = dx / L
    m = dy / L
    n = dz / L

    k = (E * A / L) * np.array([
        [ l * l,  l * m,  l * n, -l * l, -l * m, -l * n],
        [ l * m,  m * m,  m * n, -l * m, -m * m, -m * n],
        [ l * n,  m * n,  n * n, -l * n, -m * n, -n * n],
        [-l * l, -l * m, -l * n,  l * l,  l * m,  l * n],
        [-l * m, -m * m, -m * n,  l * m,  m * m,  m * n],
        [-l * n, -m * n, -n * n,  l * n,  m * n,  n * n]
    ], dtype=float)

    return L, k


def build_global_load_vector(nodes, dof_map, nodal_loads):
    ndof = 6 * len(nodes)
    f = np.zeros(ndof)

    for node_id, load_vals in nodal_loads.items():
        node_dofs = dof_map[node_id]
        for i in range(6):
            f[node_dofs[i] - 1] += load_vals[i]

    return f


def build_restrained_dofs(dof_map, supports):
    restrained = []

    for node_id, bc in supports.items():
        node_dofs = dof_map[node_id]
        for i in range(6):
            if bc[i] == 1:
                restrained.append(node_dofs[i])

    return restrained


def assemble_global_stiffness_and_fef(ndof, k_list, T_list, Qf_list, map_list):
    K_global = np.zeros((ndof, ndof), dtype=float)
    F_fef_global = np.zeros(ndof, dtype=float)

    nelem = len(k_list)

    for i in range(nelem):
        k_local = k_list[i]
        T = T_list[i]
        Qf_local = Qf_list[i]
        dof_map = map_list[i]

        edof = k_local.shape[0]

        K = T.T @ k_local @ T
        F_fef = T.T @ Qf_local

        for a in range(edof):
            A = dof_map[a] - 1
            F_fef_global[A] += F_fef[a]

            for b in range(edof):
                B = dof_map[b] - 1
                K_global[A, B] += K[a, b]

    return K_global, F_fef_global


def partition_system(K, f, u, f_fef, dof_restrained_1based):
    ndof = K.shape[0]

    restrained_dofs = sorted(int(d) - 1 for d in dof_restrained_1based)
    free_dofs = [i for i in range(ndof) if i not in restrained_dofs]

    K_ff = K[np.ix_(free_dofs, free_dofs)]
    K_fr = K[np.ix_(free_dofs, restrained_dofs)]
    K_rf = K[np.ix_(restrained_dofs, free_dofs)]
    K_rr = K[np.ix_(restrained_dofs, restrained_dofs)]

    f_f = f[free_dofs]
    f_r = f[restrained_dofs]

    u_r = u[restrained_dofs]

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
    ndof_total = len(free_dofs) + len(restrained_dofs)
    u_global = np.zeros(ndof_total)

    if u_r is None:
        u_r = np.zeros(len(restrained_dofs))

    u_global[free_dofs] = u_f
    u_global[restrained_dofs] = u_r

    return u_global


def assemble_global_forces(f_f, F_r, free_dofs, restrained_dofs):
    ndof_total = len(free_dofs) + len(restrained_dofs)
    f_global = np.zeros(ndof_total)

    f_global[free_dofs] = f_f
    f_global[restrained_dofs] = F_r

    return f_global


def create_clean_results(results, input_file):
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    nodes = results["nodes"]
    elements = results["elements"]
    supports = results["supports"]
    u_global = results["u_global"]
    f_global_complete = results["f_global_complete"]
    element_results = results["element_results"]

    sorted_nodes = sorted(nodes.keys())

    nodal_displacements = []
    support_reactions = []

    for i, node_id in enumerate(sorted_nodes):
        start = 6 * i

        nodal_displacements.append({
            "node": int(node_id),
            "ux": float(u_global[start + 0]),
            "uy": float(u_global[start + 1]),
            "uz": float(u_global[start + 2]),
            "rx": float(u_global[start + 3]),
            "ry": float(u_global[start + 4]),
            "rz": float(u_global[start + 5]),
        })

        if node_id in supports:
            support_reactions.append({
                "node": int(node_id),
                "Fx": float(f_global_complete[start + 0]),
                "Fy": float(f_global_complete[start + 1]),
                "Fz": float(f_global_complete[start + 2]),
                "Mx": float(f_global_complete[start + 3]),
                "My": float(f_global_complete[start + 4]),
                "Mz": float(f_global_complete[start + 5]),
            })

    clean_element_results = []
    for elem in element_results:
        clean_element_results.append({
            "element": int(elem["element_id"]),
            "type": elem["type"],
            "node_i": int(elem["nodes"][0]),
            "node_j": int(elem["nodes"][1]),
            "length": float(elem["L"]),
            "axial_force": float(elem["axial_force"]),
        })

    clean_results = {
        "model_name": base_name,
        "summary": {
            "number_of_nodes": len(nodes),
            "number_of_elements": len(elements),
            "total_dofs": int(results["ndof"]),
        },
        "nodal_displacements": nodal_displacements,
        "support_reactions": support_reactions,
        "element_results": clean_element_results,
        "deformed_shape_plot": f"outputs/{base_name}_deformed_shape.png",
    }

    return clean_results


def save_results_to_json(results, input_file):
    os.makedirs("outputs", exist_ok=True)

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join("outputs", base_name + "_results.json")

    clean_results = create_clean_results(results, input_file)

    with open(output_file, "w") as f:
        json.dump(clean_results, f, indent=4)

    print(f"Results saved to: {output_file}")


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

            L, k_global = k_global_3d_truss(E, A, xi, yi, zi, xj, yj, zj)

            k_local = k_global
            T = np.eye(6)
            Qf = np.zeros(6)

            dofs_i = dof_map[ni][0:3]
            dofs_j = dof_map[nj][0:3]
            elem_map = dofs_i + dofs_j

            k_list.append(k_local)
            T_list.append(T)
            Qf_list.append(Qf)
            map_list.append(elem_map)

            element_results.append({
                "element_id": elem_id,
                "type": "3d_truss",
                "nodes": [ni, nj],
                "L": L,
                "E": E,
                "A": A,
                "map": elem_map,
                "T": T,
                "k_local": k_local,
            })

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

            element_results.append({
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
            })

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
            ni, nj = elem["nodes"]
            xi, yi, zi = nodes[ni]
            xj, yj, zj = nodes[nj]

            dx = xj - xi
            dy = yj - yi
            dz = zj - zi
            L = np.sqrt(dx**2 + dy**2 + dz**2)

            l = dx / L
            m = dy / L
            n = dz / L

            u = u_elem_global

            axial_def = (
                -l * u[0] - m * u[1] - n * u[2]
                + l * u[3] + m * u[4] + n * u[5]
            )

            elem["axial_force"] = (elem["E"] * elem["A"] / L) * axial_def

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

    save_results_to_json(results, input_file)

    return results