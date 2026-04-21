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
        columns=["DOF", "Type", "Status", "Disp", "Disp Unit", "Force / Reaction"],
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


def plot_structure_3d(nodes, elements, u_global, scale=1.0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for elem_id, elem in elements.items():
        ni, nj = elem["nodes"]

        xi, yi, zi = nodes[ni]
        xj, yj, zj = nodes[nj]

        dof_i = 6 * (sorted(nodes.keys()).index(ni))
        dof_j = 6 * (sorted(nodes.keys()).index(nj))

        ui = u_global[dof_i:dof_i+3]
        uj = u_global[dof_j:dof_j+3]

        # original
        ax.plot(
            [xi, xj],
            [yi, yj],
            [zi, zj],
            "k-",
            linewidth=1.5,
        )

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
    ax.set_title(f"Original (black) and Deformed (red), scale = {scale}")
    plt.show()