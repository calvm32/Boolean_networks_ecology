import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import itertools

def phase_diagram(base_params, param_x, param_y,
                  x_range, y_range,
                  metric_key="P_max", n_grid=20,
                  n_mc=50, N0=100, steps=400):
    """
    param_x, param_y : str  — keys in base_params
    x_range, y_range : (min, max) tuples
    metric_key       : one of the scalar metrics returned by compute_metrics
    n_mc             : MC runs per grid cell (set to 1 for deterministic params)
    """
    xs = np.linspace(*x_range, n_grid)
    ys = np.linspace(*y_range, n_grid)
    Z  = np.full((n_grid, n_grid), np.nan)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            params = {**base_params, param_x: x, param_y: y}
            vals = []
            for _ in range(n_mc):
                hist = simulate(make_initial_state(), steps=steps, parameters=params)
                m    = compute_metrics(hist, N0)
                vals.append(m[metric_key])
            Z[j, i] = np.nanmean(vals)   # j=row=y-axis

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    pcm = ax.pcolormesh(xs, ys, Z, cmap="RdYlBu_r", shading="auto")
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label(metric_key, fontsize=11)

    # Add contour lines at meaningful levels (e.g., 0.2, 0.5, 0.8 for P_max)
    cs = ax.contour(xs, ys, Z, levels=5, colors="white", linewidths=0.7, alpha=0.6)
    ax.clabel(cs, inline=True, fontsize=7)

    ax.set_xlabel(param_x, fontsize=12)
    ax.set_ylabel(param_y, fontsize=12)
    ax.set_title(f"Phase Diagram: {metric_key}\n({param_x} × {param_y})", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"figures/phase_{param_x}_{param_y}_{metric_key}.pdf",
                bbox_inches="tight", dpi=300)
    plt.show()
    return xs, ys, Z