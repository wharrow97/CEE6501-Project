import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def print_dsm_results_3d(
    u_global,
    f_global_complete,
    dof_restrained_1based,
    disp_scale=1000,
    dec=4,
):
    dof_labels = ["u_x", "u_y", "u_z", "r_x", "r_y", "r_z"]
    restrained_set = set(dof_restrained_1based)

    rows = []
    for i in range(len(u_global)):
        dof_1based = i + 1
        dof_type = dof_labels[i % 6]

        if i % 6 <= 2:
            disp_val = u_global[i] * disp_scale
            disp_str = f"{disp_val:.{dec}f}"
            disp_unit = "mm"
        else:
            disp_str = f"{u_global[i]:.{dec}f}"
            disp_unit = "rad"

        load_str = f"{f_global_complete[i]:.{dec}f}"

        if dof_1based in restrained_set:
            status = "Fixed"
        else:
            status = "Free"

        rows.append([dof_1based, dof_type, status, disp_str, disp_unit, load_str])

    df = pd.DataFrame(
        rows,
        columns=["DOF", "Type", "Status", "Disp", "Unit", "Force / Reaction"],
    )

    print(df.to_string(index=False))


def print_element_results_3d(element_results, dec=4):
    rows = []

    for elem in element_results:
        rows.append(
            [
                elem["element_id"],
                elem["type"],
                elem["nodes"][0],
                elem["nodes"][1],
                round(elem["L"], dec),
                round(elem["axial_force"], dec),
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=["Element", "Type", "Node i", "Node j", "Length", "Axial Force"],
    )

    print(df.to_string(index=False))


def plot_structure_3d(nodes, elements, u_global, scale=1.0, save_path=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sorted_nodes = sorted(nodes.keys())

    for elem_id, elem in elements.items():
        ni, nj = elem["nodes"]

        xi, yi, zi = nodes[ni]
        xj, yj, zj = nodes[nj]

        i_idx = sorted_nodes.index(ni)
        j_idx = sorted_nodes.index(nj)

        ui = u_global[6*i_idx:6*i_idx+3]
        uj = u_global[6*j_idx:6*j_idx+3]

        # original
        ax.plot([xi, xj], [yi, yj], [zi, zj], "k-", linewidth=1.5)

        # deformed
        ax.plot(
            [xi + scale * ui[0], xj + scale * uj[0]],
            [yi + scale * ui[1], yj + scale * uj[1]],
            [zi + scale * ui[2], zj + scale * uj[2]],
            "r--",
            linewidth=1.5,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Deformed Shape (scale = {scale})")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def _get_local_axes_3d(p1, p2):
    """
    Returns local unit vectors ex, ey, ez for a 3D member.
    ex = along member
    ey, ez = transverse local directions
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    v = p2 - p1
    L = np.linalg.norm(v)
    ex = v / L

    # pick a reference vector that is not parallel to ex
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(ex, ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])

    ey = np.cross(ref, ex)
    ey = ey / np.linalg.norm(ey)

    ez = np.cross(ex, ey)
    ez = ez / np.linalg.norm(ez)

    return ex, ey, ez


def save_load_plot_3d(
    nodes,
    elements,
    nodal_loads,
    member_loads,
    input_file,
    nodal_scale=2.0,
    member_scale=20.0,
    num_member_arrows=4
):

    base_name = os.path.splitext(os.path.basename(input_file))[0]
    output_folder = os.path.join("outputs", base_name)
    os.makedirs(output_folder, exist_ok=True)

    save_path = os.path.join(output_folder, base_name + "_loads.png")

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for elem_id, elem_data in elements.items():
        n1, n2 = elem_data["nodes"]
        x1, y1, z1 = nodes[n1]
        x2, y2, z2 = nodes[n2]

        ax.plot(
            [x1, x2],
            [y1, y2],
            [z1, z2],
            "k-",
            linewidth=1.5
        )

    for node_id, load_vals in nodal_loads.items():
        x, y, z = nodes[node_id]

        Fx, Fy, Fz = load_vals[0], load_vals[1], load_vals[2]

        # only draw if there is a force
        if abs(Fx) > 1e-12 or abs(Fy) > 1e-12 or abs(Fz) > 1e-12:
            ax.quiver(
                x, y, z,
                Fx * nodal_scale,
                Fy * nodal_scale,
                Fz * nodal_scale,
                color="blue",
                linewidth=2.5,
                arrow_length_ratio=0.2
            )

    for elem_id, load_data in member_loads.items():
        if elem_id not in elements:
            continue

        elem = elements[elem_id]

        if elem["type"] != "3d_frame":
            continue

        n1, n2 = elem["nodes"]
        p1 = np.array(nodes[n1], dtype=float)
        p2 = np.array(nodes[n2], dtype=float)

        ex, ey, ez = _get_local_axes_3d(p1, p2)

        wy = load_data.get("wy", 0.0)
        wz = load_data.get("wz", 0.0)

        w_global = wy * ey + wz * ez

        if np.linalg.norm(w_global) < 1e-12:
            continue

        for i in range(num_member_arrows):
            t = (i + 1) / (num_member_arrows + 1)
            p = p1 + t * (p2 - p1)

            ax.quiver(
                p[0], p[1], p[2],
                w_global[0] * member_scale,
                w_global[1] * member_scale,
                w_global[2] * member_scale,
                color="green",
                linewidth=2.5,
                arrow_length_ratio=0.2
            )

    for node_id, coords in nodes.items():
        x, y, z = coords
        ax.text(x, y, z, str(node_id), fontsize=8, color="black")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Applied Loads (blue = nodal, green = member)")

    # equal-ish axis scaling
    xs = [coords[0] for coords in nodes.values()]
    ys = [coords[1] for coords in nodes.values()]
    zs = [coords[2] for coords in nodes.values()]

    x_mid = 0.5 * (min(xs) + max(xs))
    y_mid = 0.5 * (min(ys) + max(ys))
    z_mid = 0.5 * (min(zs) + max(zs))

    max_range = max(
        max(xs) - min(xs),
        max(ys) - min(ys),
        max(zs) - min(zs)
    ) / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Loads plot saved to: {save_path}")