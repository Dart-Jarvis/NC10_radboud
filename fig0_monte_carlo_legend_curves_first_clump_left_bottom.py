#!/usr/bin/env python3
"""
Monte Carlo uncertainty propagation + plotting for fig_bulk0 / fig_clump0.

This script combines the previous two-script workflow into one script:

    1. Load the model functions from MODEL_PY.
    2. For each selected series, run the central model trajectory.
    3. Randomly sample rev1-rev3 plus five forward gamma factors and run Monte Carlo model trajectories.
    4. Calculate uncertainty envelopes from Monte Carlo percentiles.
    5. Plot the model envelope, central model line, and experimental data.
    6. Save trajectory/cache CSV files and final PDF figures.

Important:
    - This script does NOT use finite-difference error propagation.
    - The uncertainty region is calculated directly from Monte Carlo samples.
    - Experimental labels "AOM_P_LS", "AOM_Wegner_LS", and "AOM_Wegener_LS"
      are combined as the plotted series "AOM_LS".

Run:
    python fig0_monte_carlo_model_and_plot.py
"""

# =============================================================================
# USER SETTINGS
# =============================================================================

F420=True
MODEL_PY = "multistep_model_step2_primary_secondary.py"
if F420==False:
    MODEL_PY = "multistep_model_no_f420.py"
DATA_CSV = "isotope_data.csv"

OUT_PREFIX = "fig0_monte_carlo"

INCLUDE_ANME2D = True
INCLUDE_AOM_P = True
INCLUDE_AOM_LS = True

# Optional extra series with no experimental data points.
# These series are configured below to display only their central model lines.
INCLUDE_MODEL_ONLY = True
MODEL_ONLY_SERIES_NAME = "Irreversible model"

INCLUDE_MODEL_ONLY_2 = True
MODEL_ONLY_SERIES_NAME_2 = "Near-equilibrium model"

INCLUDE_MODEL_ONLY_3 = True
MODEL_ONLY_SERIES_NAME_3 = "Intermediate-reversibility model"

SERIES_TO_RUN = None   # Example: ["AOM_LS"] or None for all included series

# -----------------------------------------------------------------------------
# Monte Carlo settings
# -----------------------------------------------------------------------------

RANDOM_SEED = 12345
N_MONTE_CARLO = 500

# Monte Carlo uncertainty intervals to plot.
# For a normal distribution:
#   68.27% ≈ ±1 sigma
#   95.45% ≈ ±2 sigma
# The script will draw the 2-sigma region first with a lighter color,
# then draw the 1-sigma region on top with a deeper color.
MC_INTERVALS_TO_PLOT = {
    "1sigma": 68.27,
    "2sigma": 95.45,
}

# Parameter sampling mode:
#   "truncated_normal" draws from a normal distribution centered at the
#   central value with standard deviation = 1-sigma uncertainty, but samples
#   are drawn only inside the allowed bounds. This avoids artificial pile-up
#   at the bounds.
PARAMETER_SAMPLING_MODE = "truncated_normal"

# Save every Monte Carlo trajectory to CSV. This is useful for debugging, but the
# CSV files can become large when N_MONTE_CARLO is high.
SAVE_MC_TRAJECTORIES = True
SAVE_MC_REGIONS = True

# If True, draw all Monte Carlo trajectories lightly behind the central model.
# This can make plots busy; the percentile region is usually cleaner.
PLOT_INDIVIDUAL_MC_TRAJECTORIES = False
MC_TRAJECTORY_ALPHA = 0.04
MC_TRAJECTORY_LINEWIDTH = 0.7

# -----------------------------------------------------------------------------
# Series definitions
# -----------------------------------------------------------------------------

# These names are the internal model/plot series names.
SERIES_TO_PLOT = [
    "ANME2d",
    "AOM_P",
    "AOM_LS",
    MODEL_ONLY_SERIES_NAME,
    MODEL_ONLY_SERIES_NAME_2,
    MODEL_ONLY_SERIES_NAME_3,
]

# Experimental labels in isotope_data.csv that should be plotted for each series.
# AOM_LS combines both low-sulfate datasets.
SERIES_DATA_LABELS = {
    "ANME2d": ["3"],
    "AOM_P": ["AOM_P"],
    "AOM_LS": ["AOM_P_LS"],
    MODEL_ONLY_SERIES_NAME: [],
    MODEL_ONLY_SERIES_NAME_2: ["AOM_Wegener_LS"],
    MODEL_ONLY_SERIES_NAME_3: ["AOM_Wegener","AOM_ono"],
}

# Central reversibility values: [rev1, rev2, rev3]
SERIES_REVS = {
    "ANME2d": [0.34, 0.95, 0.04],
    "AOM_P":  [0.73, 0.65, 0.95],
    "AOM_LS": [0.96, 0.99, 0.99],
    MODEL_ONLY_SERIES_NAME: [0.0, 0.0, 0.0],
    MODEL_ONLY_SERIES_NAME_2: [0.80, 0.95, 0.95],
    MODEL_ONLY_SERIES_NAME_3: [0.50, 0.80, 0.50],
}

if F420==False:
    SERIES_REVS = {
    "ANME2d": [0.34, 0.75, 0.04],
    "AOM_P":  [0.61, 0.95, 0.04],
    "AOM_LS": [0.90, 0.95, 0.95],
    MODEL_ONLY_SERIES_NAME: [0.0, 0.0, 0.0],
    MODEL_ONLY_SERIES_NAME_2: [0.99, 0.99, 0.99],
    MODEL_ONLY_SERIES_NAME_3: [0.50, 0.50, 0.50],
    }

# 1-sigma uncertainties for [rev1, rev2, rev3]
SERIES_SIGMA_REVS = {
    "ANME2d": [0.02, 0.02, 0.02],
    "AOM_P":  [0.02, 0.02, 0.02],
    "AOM_LS": [0.015, 0.004, 0.004],
    MODEL_ONLY_SERIES_NAME: [0.0, 0.0, 0.0],
    MODEL_ONLY_SERIES_NAME_2: [0.02, 0.02, 0.02],
    MODEL_ONLY_SERIES_NAME_3: [0.02, 0.02, 0.02],
}

# Shared central forward gamma factors and their 1-sigma uncertainties.
# Replace these central values and uncertainties with the outputs from your
# best-fit search. The current Step-2 defaults preserve the previous aggregate
# gamma values by assigning them to both positional pathways.
#
# Step 2, 13C-D:
#   p  = D experiences the primary isotope effect
#   s  = D experiences the secondary isotope effect
# Step 2, D-D:
#   pp = both D atoms experience primary effects
#   ps = one D experiences a primary effect and one a secondary effect
# Step 3:
#   gammaCDff3 controls the forward 13C-D effect for CHO-MFR -> CO2.
GAMMA_CENTRAL = {
    "gammaCDff2p":  0.9749,
    "gammaCDff2s":  1.0000,
    "gammaDDff2pp": 0.9256,
    "gammaDDff2ps": 0.9255,
    "gammaCDff3":   1.0000,
}

if F420==False:
    GAMMA_CENTRAL = {
        "gammaCDff2p":  0.9868,
        "gammaCDff2s":  0.9502,
        "gammaDDff2pp": 0.9872,
        "gammaDDff2ps": 0.9518,
        "gammaCDff3":   0.9502,
}


GAMMA_SIGMA = {
    "gammaCDff2p":  0.005,
    "gammaCDff2s":  0.000,
    "gammaDDff2pp": 0.005,
    "gammaDDff2ps": 0.005,
    "gammaCDff3":   0.000,
}

if F420==False:
    GAMMA_SIGMA = {
        "gammaCDff2p":  0.005,
        "gammaCDff2s":  0.005,
        "gammaDDff2pp": 0.005,
        "gammaDDff2ps": 0.005,
        "gammaCDff3":   0.005,
    }

# Bounds for sampled parameters. These match the search ranges:
# all CD gamma factors: 0.95-1.00; all DD gamma factors: 0.90-1.00.
REV_LOWER = 0.00
REV_UPPER = 0.999
GAMMA_BOUNDS = {
    "gammaCDff2p":  (0.95, 1.00),
    "gammaCDff2s":  (0.95, 1.00),
    "gammaDDff2pp": (0.90, 1.00),
    "gammaDDff2ps": (0.90, 1.00),
    "gammaCDff3":   (0.95, 1.00),
}

if F420==False:
    GAMMA_BOUNDS = {
        "gammaCDff2p":  (0.90, 1.00),
        "gammaCDff2s":  (0.90, 1.00),
        "gammaDDff2pp": (0.90, 1.00),
        "gammaDDff2ps": (0.90, 1.00),
        "gammaCDff3":   (0.90, 1.00),
    }


# -----------------------------------------------------------------------------
# ODE time and f-grid settings
# -----------------------------------------------------------------------------

DEFAULT_TMAX = 500.0
SERIES_TMAX = {
    "ANME2d": 500.0,
    "AOM_P": 500.0,
    "AOM_LS": 500.0,
    MODEL_ONLY_SERIES_NAME: 500.0,
    MODEL_ONLY_SERIES_NAME_2: 500.0,
    MODEL_ONLY_SERIES_NAME_3: 500.0,
}

USE_EVENT_STOP = True

# If USE_MANUAL_F_MIN[series] is False, the script will try to use the smallest
# valid experimental f value from the matching CSV labels. If f is missing, it
# falls back to MANUAL_F_MIN.
MANUAL_F_MIN = {
    "ANME2d": 0.30,
    "AOM_P": 0.20,
    "AOM_LS": 0.90,
    MODEL_ONLY_SERIES_NAME: 0.40,
    MODEL_ONLY_SERIES_NAME_2: 0.75,
    MODEL_ONLY_SERIES_NAME_3: 0.20,
}

USE_MANUAL_F_MIN = {
    "ANME2d": True,
    "AOM_P": True,
    "AOM_LS": True,
    MODEL_ONLY_SERIES_NAME: True,
    MODEL_ONLY_SERIES_NAME_2: True,
    MODEL_ONLY_SERIES_NAME_3: True,
}

F_MARGIN = 0.005
NGRID_F = 120

ODE_METHOD = "LSODA"
ODE_ATOL = 1e-9
ODE_RTOL = 1e-6

# -----------------------------------------------------------------------------
# User-editable plot labels and styles
# -----------------------------------------------------------------------------
def model_labels(s):
    label=r"R$_1$="+str(SERIES_REVS[s][0])+r"$\pm$"+str(SERIES_SIGMA_REVS[s][0])+r", R$_2$="+str(SERIES_REVS[s][1])+r"$\pm$"+str(SERIES_SIGMA_REVS[s][1])+r", R$_3$="+str(SERIES_REVS[s][2])+r"$\pm$"+str(SERIES_SIGMA_REVS[s][2])
    return label

labels={}
for key in SERIES_REVS:
    labels[key]=model_labels(key)

SERIES_DISPLAY_LABELS = {
    "ANME2d": "N-AOM (ANME, this study)",
    "AOM_P": "S-AOM (high sulfate 1)",
    "AOM_LS": "S-AOM (low sulfate)",
    MODEL_ONLY_SERIES_NAME: "Model-only series",
    MODEL_ONLY_SERIES_NAME_2: "Near-equilibrium model",
    MODEL_ONLY_SERIES_NAME_3: "S-AOM (high sulfate 2)",
}

# Set any of these to None to omit that legend item.
SERIES_REGION_LABELS = {
    "ANME2d": None,
    "AOM_P": None,
    "AOM_LS": None,
    MODEL_ONLY_SERIES_NAME: None,
    MODEL_ONLY_SERIES_NAME_2: None,
    MODEL_ONLY_SERIES_NAME_3: None,
}

SERIES_MODEL_LABELS = {
    "ANME2d": labels["ANME2d"],
    "AOM_P": labels["AOM_P"],
    "AOM_LS": labels["AOM_LS"],
    MODEL_ONLY_SERIES_NAME: r"Irreversible model, R$_1$=R$_2$=R$_3$=0.0",
    MODEL_ONLY_SERIES_NAME_2: labels[MODEL_ONLY_SERIES_NAME_2],
    MODEL_ONLY_SERIES_NAME_3: labels[MODEL_ONLY_SERIES_NAME_3],
}

SERIES_DATA_LABELS_FOR_LEGEND = {
    "ANME2d": "N-AOM (ANME, this study)",
    "AOM_P": "S-AOM (high sulfate 2)",
    "AOM_LS": "S-AOM (low sulfate 2)",
    MODEL_ONLY_SERIES_NAME: "Model-only series",
    MODEL_ONLY_SERIES_NAME_2: "S-AOM (low sulfate 1)",
    MODEL_ONLY_SERIES_NAME_3: "S-AOM (high sulfate 1)",
}

# Any Matplotlib color name, hex code, or RGB/RGBA tuple works.
SERIES_COLORS = {
    "ANME2d": "tab:blue",
    "AOM_P": "tab:orange",
    "AOM_LS": "tab:gray",
    MODEL_ONLY_SERIES_NAME: "black",
    MODEL_ONLY_SERIES_NAME_2: "tab:green",
    MODEL_ONLY_SERIES_NAME_3: "tab:purple",
}

# Set a value to None to use SERIES_COLORS for that series.
SERIES_REGION_COLORS = {
    "ANME2d": None,
    "AOM_P": None,
    "AOM_LS": None,
    MODEL_ONLY_SERIES_NAME: None,
    MODEL_ONLY_SERIES_NAME_2: None,
    MODEL_ONLY_SERIES_NAME_3: None,
}

SERIES_LINESTYLES = {
    "ANME2d": "-",
    "AOM_P": "-.",
    "AOM_LS": "--",
    MODEL_ONLY_SERIES_NAME: "-",
    MODEL_ONLY_SERIES_NAME_2: "--",
    MODEL_ONLY_SERIES_NAME_3: "-.",
}

SERIES_LINEWIDTHS = {
    "ANME2d": 2.5,
    "AOM_P": 2.5,
    "AOM_LS": 2.5,
    MODEL_ONLY_SERIES_NAME: 2.5,
    MODEL_ONLY_SERIES_NAME_2: 2.5,
    MODEL_ONLY_SERIES_NAME_3: 2.5,
}

SERIES_MARKERS = {
    "ANME2d": "o",
    "AOM_P": "s",
    "AOM_LS": "^",
    MODEL_ONLY_SERIES_NAME: "D",
    MODEL_ONLY_SERIES_NAME_2: "<",
    MODEL_ONLY_SERIES_NAME_3: "v",
}

SERIES_MARKER_SIZES = {
    "ANME2d": 8,
    "AOM_P": 8,
    "AOM_LS": 8,
    MODEL_ONLY_SERIES_NAME: 8,
    MODEL_ONLY_SERIES_NAME_2: 8,
    MODEL_ONLY_SERIES_NAME_3: 8,
}

SERIES_MARKER_FACE_COLORS = {
    "ANME2d": None,
    "AOM_P": None,
    "AOM_LS": None,
    MODEL_ONLY_SERIES_NAME: None,
    MODEL_ONLY_SERIES_NAME_2: None,
    MODEL_ONLY_SERIES_NAME_3: None,
}

SERIES_MARKER_EDGE_COLORS = {
    "ANME2d": "black",
    "AOM_P": "black",
    "AOM_LS": "black",
    MODEL_ONLY_SERIES_NAME: "black",
    MODEL_ONLY_SERIES_NAME_2: "black",
    MODEL_ONLY_SERIES_NAME_3: "black",
}

SERIES_MARKER_EDGE_WIDTHS = {
    "ANME2d": 1.0,
    "AOM_P": 1.0,
    "AOM_LS": 1.0,
    MODEL_ONLY_SERIES_NAME: 1.0,
    MODEL_ONLY_SERIES_NAME_2: 1.0,
    MODEL_ONLY_SERIES_NAME_3: 1.0,
}

# Region transparency. Larger alpha = deeper/darker color.
# The 2-sigma region is plotted first and lighter.
# The 1-sigma region is plotted on top and deeper.
REGION_ALPHA_1SIGMA = 0.4
REGION_ALPHA_2SIGMA = 0.2

# Legend labels for uncertainty regions.
# Set any value to None if you do not want that region in the legend.
SERIES_REGION_LABELS_1SIGMA = {
    "ANME2d": None,
    "AOM_P": None,
    "AOM_LS": None,
    MODEL_ONLY_SERIES_NAME: None,
    MODEL_ONLY_SERIES_NAME_2: None,
    MODEL_ONLY_SERIES_NAME_3: None,
}

SERIES_REGION_LABELS_2SIGMA = {
    "ANME2d": None,
    "AOM_P": None,
    "AOM_LS": None,
    MODEL_ONLY_SERIES_NAME: None,
    MODEL_ONLY_SERIES_NAME_2: None,
    MODEL_ONLY_SERIES_NAME_3: None,
}

# Control whether Monte Carlo trajectories and uncertainty regions are drawn.
# The Monte Carlo calculations and CSV outputs are still produced even when
# plotting is disabled, but only the deterministic central line is displayed.
SERIES_PLOT_UNCERTAINTY = {
    "ANME2d": True,
    "AOM_P": True,
    "AOM_LS": True,
    MODEL_ONLY_SERIES_NAME: False,
    MODEL_ONLY_SERIES_NAME_2: True,
    MODEL_ONLY_SERIES_NAME_3: True,
}

EXCLUDE_INITIAL_DATA_FROM_PLOT = False

XLIM_BULK = None
YLIM_BULK = None
XLIM_CLUMP = None
YLIM_CLUMP = None

# =============================================================================
# END USER SETTINGS
# =============================================================================

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import truncnorm
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

font_labels = {'family': 'sans serif',
               'color':  'black',
               'weight': 'normal',
               'size': 24,
                }

TRANSFORMED_COLS = ["ln(c/c0)", "ln(d/d0)", "DeltaCD", "DeltaDD"]
PARAMETER_COLS = ["rev1", "rev2", "rev3", "gammaCDff2p", "gammaCDff2s", "gammaDDff2pp", "gammaDDff2ps", "gammaCDff3"]


def style_value(style_dict, series, default=None):
    value = style_dict.get(series, default)
    return default if value is None else value


def display_label(series):
    return SERIES_DISPLAY_LABELS.get(series, series)


def add_legend_sorted_by_rev1(ax, **legend_kwargs):
    """Create a legend with curves first and data points second.

    Within both groups, series are ordered from low to high central R1.
    Uncertainty-region entries, when enabled, are treated as curve-related
    entries and remain ahead of the experimental data entries.
    """
    handles, legend_labels = ax.get_legend_handles_labels()
    if not handles:
        return None

    # Map each possible legend label to its series and entry type.
    # group = 0: modeled curves / uncertainty regions
    # group = 1: experimental data points
    label_metadata = {}
    for series in SERIES_REVS:
        curve_labels = [
            SERIES_REGION_LABELS_2SIGMA.get(
                series, f"{display_label(series)} ±2σ MC region"
            ),
            SERIES_REGION_LABELS_1SIGMA.get(
                series, f"{display_label(series)} ±1σ MC region"
            ),
            SERIES_MODEL_LABELS.get(
                series, f"{display_label(series)} central model"
            ),
        ]
        data_label = SERIES_DATA_LABELS_FOR_LEGEND.get(
            series, f"{display_label(series)} data"
        )

        for label in curve_labels:
            if label is not None:
                label_metadata[label] = (series, 0)
        if data_label is not None:
            label_metadata[data_label] = (series, 1)

    indexed_entries = list(enumerate(zip(handles, legend_labels)))

    def sort_key(item):
        original_index, (_handle, label) = item
        metadata = label_metadata.get(label)
        if metadata is None:
            # Keep unrecognized entries after all known curve and data entries.
            return (2, float("inf"), original_index)

        series, group = metadata
        return (group, float(SERIES_REVS[series][0]), original_index)

    indexed_entries.sort(key=sort_key)
    sorted_handles = [entry[1][0] for entry in indexed_entries]
    sorted_labels = [entry[1][1] for entry in indexed_entries]

    return ax.legend(sorted_handles, sorted_labels, **legend_kwargs)


def load_model_prelude(model_py):
    model_py = Path(model_py)
    if not model_py.exists():
        raise FileNotFoundError(
            f"Missing model file: {model_py}\n"
            "Set MODEL_PY to the path of your main model script, for example "
            "MODEL_PY = 'multistep_model_step2_primary_secondary.py'."
        )

    source = model_py.read_text()
    marker = "if len(rev1_list) != len(rev2_list)"
    if marker not in source:
        raise RuntimeError(
            "Could not find the top-level simulation marker in the model file.\n"
            "The script expects your model file to define dfdt(), process(), R0, "
            "t_lower, and isotope-factor variables before the main simulation loop."
        )

    prelude = source.split(marker)[0]
    module = type(sys)("aom_model")
    module.__file__ = str(model_py)
    sys.modules["aom_model"] = module
    exec(compile(prelude, str(model_py), "exec"), module.__dict__)
    return module


def validate_model_gamma_interface(model):
    """Check that the loaded model supports all five forward gamma variables."""
    required = [
        "gammaCDff2p", "gammaCDff2s", "gammaDDff2pp", "gammaDDff2ps",
        "gammaCDff3", "a2cff", "a2dffp", "a2dffs", "a3cff", "a3dff",
    ]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise AttributeError(
            "The selected MODEL_PY is not the primary/secondary Step-2 model. "
            "Missing variables: " + ", ".join(missing)
        )


def set_forward_gammas(model, gammas):
    """Update all five sampled forward gamma factors and derived alphas."""
    validate_model_gamma_interface(model)

    for name in GAMMA_CENTRAL:
        if name not in gammas:
            raise KeyError(f"Missing sampled gamma value: {name}")
        setattr(model, name, float(gammas[name]))

    # Step 2: pathway-specific 13C-D and D-D forward fractionation factors.
    model.a2cdffp = model.gammaCDff2p * model.a2cff * model.a2dffp
    model.a2cdffs = model.gammaCDff2s * model.a2cff * model.a2dffs
    model.a2ddffpp = model.gammaDDff2pp * model.a2dffp**2
    model.a2ddffps = model.gammaDDff2ps * model.a2dffp * model.a2dffs

    # Step 3: forward 13C-D clumped fractionation factor.
    model.a3cdff = model.gammaCDff3 * model.a3cff * model.a3dff


def run_model_curve(model, revs, gammas, f_grid, f_stop, tmax):
    """Run one ODE trajectory and interpolate transformed outputs onto f_grid."""
    set_forward_gammas(model, gammas)

    rev1, rev2, rev3 = [float(x) for x in revs]
    k = [1.0, rev1]
    tot0 = float(np.sum(model.R0[12:17]))

    def reach_min_f(_t, R, *unused_args):
        return float(np.sum(R[12:17]) / tot0 - f_stop)

    reach_min_f.terminal = True
    reach_min_f.direction = -1
    events = reach_min_f if USE_EVENT_STOP else None

    t0 = time.time()
    sol = solve_ivp(
        model.dfdt,
        (model.t_lower, tmax),
        model.R0,
        args=(k, rev2, rev3),
        method=ODE_METHOD,
        atol=ODE_ATOL,
        rtol=ODE_RTOL,
        dense_output=True,
        events=events,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    if USE_EVENT_STOP and sol.t_events and len(sol.t_events[0]) > 0:
        t_end = float(sol.t_events[0][0])
        event_reached = True
    else:
        t_end = float(sol.t[-1])
        event_reached = False

    t_eval = np.linspace(model.t_lower, t_end, len(f_grid))
    y_eval = sol.sol(t_eval)

    fake_sol = SimpleNamespace(t=t_eval, y=y_eval)
    _tot, f_model, d13C, dD, D13CH3D, D12CH2D2 = model.process(fake_sol)

    f_model = np.asarray(f_model, dtype=float)
    d13C = np.asarray(d13C, dtype=float)
    dD = np.asarray(dD, dtype=float)
    D13CH3D = np.asarray(D13CH3D, dtype=float)
    D12CH2D2 = np.asarray(D12CH2D2, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_c_c0 = np.log((d13C + 1000.0) / (d13C[0] + 1000.0))
        ln_d_d0 = np.log((dD + 1000.0) / (dD[0] + 1000.0))

    delta_cd = D13CH3D - D13CH3D[0]
    delta_dd = D12CH2D2 - D12CH2D2[0]

    order = np.argsort(f_model)
    f_sorted = f_model[order]
    f_unique, unique_idx = np.unique(f_sorted, return_index=True)

    if len(f_unique) < 2:
        raise RuntimeError("Model output has fewer than two unique f values; cannot interpolate.")

    def interp_to_grid(y):
        y_sorted = np.asarray(y, dtype=float)[order][unique_idx]
        fn = interp1d(
            f_unique,
            y_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=(y_sorted[0], y_sorted[-1]),
        )
        return fn(f_grid)

    elapsed = time.time() - t0

    out = pd.DataFrame({
        "f": f_grid,
        "ln(c/c0)": interp_to_grid(ln_c_c0),
        "ln(d/d0)": interp_to_grid(ln_d_d0),
        "DeltaCD": interp_to_grid(delta_cd),
        "DeltaDD": interp_to_grid(delta_dd),
    })

    return out, elapsed, t_end, event_reached


def build_series_specs():
    specs = {}

    if INCLUDE_ANME2D:
        specs["ANME2d"] = {
            "data_labels": SERIES_DATA_LABELS["ANME2d"],
            "revs": SERIES_REVS["ANME2d"],
            "rev_sigmas": SERIES_SIGMA_REVS["ANME2d"],
        }

    if INCLUDE_AOM_P:
        specs["AOM_P"] = {
            "data_labels": SERIES_DATA_LABELS["AOM_P"],
            "revs": SERIES_REVS["AOM_P"],
            "rev_sigmas": SERIES_SIGMA_REVS["AOM_P"],
        }

    if INCLUDE_AOM_LS:
        specs["AOM_LS"] = {
            "data_labels": SERIES_DATA_LABELS["AOM_LS"],
            "revs": SERIES_REVS["AOM_LS"],
            "rev_sigmas": SERIES_SIGMA_REVS["AOM_LS"],
        }

    if INCLUDE_MODEL_ONLY:
        specs[MODEL_ONLY_SERIES_NAME] = {
            "data_labels": SERIES_DATA_LABELS.get(MODEL_ONLY_SERIES_NAME, []),
            "revs": SERIES_REVS[MODEL_ONLY_SERIES_NAME],
            "rev_sigmas": SERIES_SIGMA_REVS[MODEL_ONLY_SERIES_NAME],
        }

    if INCLUDE_MODEL_ONLY_2:
        specs[MODEL_ONLY_SERIES_NAME_2] = {
            "data_labels": SERIES_DATA_LABELS.get(MODEL_ONLY_SERIES_NAME_2, []),
            "revs": SERIES_REVS[MODEL_ONLY_SERIES_NAME_2],
            "rev_sigmas": SERIES_SIGMA_REVS[MODEL_ONLY_SERIES_NAME_2],
        }

    if INCLUDE_MODEL_ONLY_3:
        specs[MODEL_ONLY_SERIES_NAME_3] = {
            "data_labels": SERIES_DATA_LABELS.get(MODEL_ONLY_SERIES_NAME_3, []),
            "revs": SERIES_REVS[MODEL_ONLY_SERIES_NAME_3],
            "rev_sigmas": SERIES_SIGMA_REVS[MODEL_ONLY_SERIES_NAME_3],
        }

    if SERIES_TO_RUN is not None:
        specs = {k: v for k, v in specs.items() if k in SERIES_TO_RUN}

    if not specs:
        raise ValueError("No series selected.")

    return specs


def determine_f_min(data, series_name, data_labels):
    use_manual = USE_MANUAL_F_MIN.get(series_name, False)

    if not use_manual:
        labels = [str(x).strip() for x in data_labels]
        df = data[data["label"].astype(str).str.strip().isin(labels)].copy()
        if "f" in df.columns:
            valid_f = pd.to_numeric(df["f"], errors="coerce").dropna()
            if len(valid_f) > 0:
                return float(valid_f.min()), "data"

    return float(MANUAL_F_MIN[series_name]), "manual"


def sample_truncated_normal(rng, mean, sigma, lower, upper):
    """
    Draw one value from a truncated normal distribution without clipping.

    This avoids artificial pile-up at the parameter bounds. If sigma is zero
    or negative, the central value is returned after checking it against bounds.
    """
    mean = float(mean)
    sigma = float(sigma)
    lower = float(lower)
    upper = float(upper)

    if lower > upper:
        raise ValueError(f"Invalid bounds: lower={lower}, upper={upper}")

    if sigma <= 0:
        if not (lower <= mean <= upper):
            raise ValueError(
                f"Central value {mean} is outside bounds [{lower}, {upper}] "
                "and sigma <= 0, so it cannot be sampled."
            )
        return mean

    a = (lower - mean) / sigma
    b = (upper - mean) / sigma
    return float(truncnorm.rvs(a, b, loc=mean, scale=sigma, random_state=rng))


def sample_parameters(rng, central_revs, sigma_revs):
    if PARAMETER_SAMPLING_MODE != "truncated_normal":
        raise ValueError(f"Unsupported PARAMETER_SAMPLING_MODE: {PARAMETER_SAMPLING_MODE}")

    revs = [
        sample_truncated_normal(rng, mean, sigma, REV_LOWER, REV_UPPER)
        for mean, sigma in zip(central_revs, sigma_revs)
    ]

    gammas = {}
    for name, mean in GAMMA_CENTRAL.items():
        lower, upper = GAMMA_BOUNDS[name]
        gammas[name] = sample_truncated_normal(
            rng,
            mean,
            GAMMA_SIGMA[name],
            lower,
            upper,
        )

    return revs, gammas


def add_metadata(curve, series_name, run_name, sample_id, revs, gammas,
                 elapsed, t_end, tmax, event_reached, f_stop):
    out = curve.copy()
    out.insert(0, "series", series_name)
    out.insert(1, "run_name", run_name)
    out.insert(2, "sample_id", sample_id)
    out.insert(3, "rev1", float(revs[0]))
    out.insert(4, "rev2", float(revs[1]))
    out.insert(5, "rev3", float(revs[2]))

    insert_at = 6
    for name in GAMMA_CENTRAL:
        out.insert(insert_at, name, float(gammas[name]))
        insert_at += 1

    out["solve_seconds"] = elapsed
    out["t_end"] = t_end
    out["tmax"] = tmax
    out["event_reached"] = event_reached
    out["f_stop"] = f_stop
    return out


def run_monte_carlo_for_series(model, series_name, spec, data, rng):
    f_min, f_source = determine_f_min(data, series_name, spec["data_labels"])
    f_stop = max(0.0, f_min - F_MARGIN)
    f_grid = np.linspace(f_min, 1.0, NGRID_F)
    tmax = float(SERIES_TMAX.get(series_name, DEFAULT_TMAX))

    central_revs = [float(x) for x in spec["revs"]]
    sigma_revs = [float(x) for x in spec["rev_sigmas"]]

    print(f"\nRunning Monte Carlo for {series_name}...")
    print(f"  data labels = {spec['data_labels']}")
    print(f"  f_min = {f_min:.5f} ({f_source}); f_stop = {f_stop:.5f}")
    print(f"  central revs = {central_revs}")
    print(f"  sigma revs   = {sigma_revs}")
    print(f"  N_MC = {N_MONTE_CARLO}; tmax = {tmax}")

    central_gammas = {name: float(value) for name, value in GAMMA_CENTRAL.items()}
    print(f"  central gammas = {central_gammas}")
    print(f"  sigma gammas   = {GAMMA_SIGMA}")

    # Central run.
    central_curve, elapsed, t_end, event_reached = run_model_curve(
        model=model,
        revs=central_revs,
        gammas=central_gammas,
        f_grid=f_grid,
        f_stop=f_stop,
        tmax=tmax,
    )
    central_out = add_metadata(
        central_curve, series_name, "central", 0, central_revs, central_gammas,
        elapsed, t_end, tmax, event_reached, f_stop
    )
    print(f"  central done in {elapsed:.2f} s")

    sample_rows = []
    failed = []
    total_start = time.time()

    for i in range(1, N_MONTE_CARLO + 1):
        revs, gammas = sample_parameters(rng, central_revs, sigma_revs)
        try:
            curve, elapsed, t_end, event_reached = run_model_curve(
                model=model,
                revs=revs,
                gammas=gammas,
                f_grid=f_grid,
                f_stop=f_stop,
                tmax=tmax,
            )
            sample_rows.append(add_metadata(
                curve, series_name, "mc", i, revs, gammas,
                elapsed, t_end, tmax, event_reached, f_stop
            ))
        except Exception as exc:
            failed_row = {
                "series": series_name,
                "sample_id": i,
                "rev1": revs[0],
                "rev2": revs[1],
                "rev3": revs[2],
                "error": str(exc),
            }
            failed_row.update(gammas)
            failed.append(failed_row)

        if i % 25 == 0 or i == N_MONTE_CARLO:
            print(f"  completed {i}/{N_MONTE_CARLO} samples; failed = {len(failed)}")

    if len(sample_rows) < 3:
        raise RuntimeError(
            f"Only {len(sample_rows)} Monte Carlo samples succeeded for {series_name}; "
            "cannot calculate a reliable uncertainty envelope."
        )

    mc_df = pd.concat(sample_rows, ignore_index=True)
    all_df = pd.concat([central_out, mc_df], ignore_index=True)

    region = pd.DataFrame({"f": f_grid})
    for col in TRANSFORMED_COLS:
        pivot = mc_df.pivot(index="sample_id", columns="f", values=col)
        # Ensure the columns follow f_grid order exactly.
        pivot = pivot.reindex(columns=f_grid)
        arr = pivot.to_numpy()

        # Always save the Monte Carlo median and deterministic central model.
        region[f"{col}_median"] = np.nanpercentile(arr, 50.0, axis=0)
        region[f"{col}_central"] = central_curve[col].to_numpy()

        # Save each requested percentile interval.
        for interval_name, interval_percent in MC_INTERVALS_TO_PLOT.items():
            low_q = 50.0 - float(interval_percent) / 2.0
            high_q = 50.0 + float(interval_percent) / 2.0
            region[f"{col}_lo_{interval_name}"] = np.nanpercentile(arr, low_q, axis=0)
            region[f"{col}_hi_{interval_name}"] = np.nanpercentile(arr, high_q, axis=0)

        # Backward-compatible aliases: _lo and _hi mean 1-sigma region.
        if "1sigma" in MC_INTERVALS_TO_PLOT:
            region[f"{col}_lo"] = region[f"{col}_lo_1sigma"]
            region[f"{col}_hi"] = region[f"{col}_hi_1sigma"]

    if SAVE_MC_TRAJECTORIES:
        traj_path = f"{OUT_PREFIX}_{series_name}_mc_trajectories.csv"
        all_df.to_csv(traj_path, index=False)
        print(f"  saved trajectories: {traj_path}")

    if SAVE_MC_REGIONS:
        region_path = f"{OUT_PREFIX}_{series_name}_mc_region.csv"
        region.to_csv(region_path, index=False)
        print(f"  saved region: {region_path}")

    if failed:
        fail_path = f"{OUT_PREFIX}_{series_name}_failed_samples.csv"
        pd.DataFrame(failed).to_csv(fail_path, index=False)
        print(f"  saved failed-sample log: {fail_path}")

    settings = {
        "series": series_name,
        "data_labels": ";".join(spec["data_labels"]),
        "rev1": central_revs[0],
        "rev2": central_revs[1],
        "rev3": central_revs[2],
        "sigma_rev1": sigma_revs[0],
        "sigma_rev2": sigma_revs[1],
        "sigma_rev3": sigma_revs[2],
        **{name: GAMMA_CENTRAL[name] for name in GAMMA_CENTRAL},
        **{f"sigma_{name}": GAMMA_SIGMA[name] for name in GAMMA_CENTRAL},
        **{f"lower_{name}": GAMMA_BOUNDS[name][0] for name in GAMMA_CENTRAL},
        **{f"upper_{name}": GAMMA_BOUNDS[name][1] for name in GAMMA_CENTRAL},
        "n_monte_carlo_requested": N_MONTE_CARLO,
        "n_monte_carlo_succeeded": len(sample_rows),
        "n_monte_carlo_failed": len(failed),
        "mc_intervals_to_plot": ";".join(
            f"{name}:{percent}" for name, percent in MC_INTERVALS_TO_PLOT.items()
        ),
        "sampling_mode": PARAMETER_SAMPLING_MODE,
        "random_seed": RANDOM_SEED,
        "model_py": MODEL_PY,
        "data_csv": DATA_CSV,
        "tmax": tmax,
        "ngrid_f": NGRID_F,
        "ode_method": ODE_METHOD,
        "ode_atol": ODE_ATOL,
        "ode_rtol": ODE_RTOL,
        "f_min": f_min,
        "f_source": f_source,
        "f_stop": f_stop,
        "use_event_stop": USE_EVENT_STOP,
        "series_runtime_seconds": time.time() - total_start,
    }

    return central_out, mc_df, region, settings


def draw_parametric_region(ax, x_lo, y_lo, x_hi, y_hi, label=None, alpha=0.18, color=None):
    x_poly = np.r_[x_lo, x_hi[::-1]]
    y_poly = np.r_[y_lo, y_hi[::-1]]
    ax.fill(x_poly, y_poly, alpha=alpha, label=label, color=color, linewidth=0)


def get_valid_data_for_series(data, series):
    csv_labels = SERIES_DATA_LABELS.get(series, [series])
    csv_labels = [str(x).strip() for x in csv_labels]
    df = data[data["label"].astype(str).str.strip().isin(csv_labels)].copy()

    if df.empty:
        return pd.DataFrame()

    optional_numeric_cols = ["f", "lncse", "lndse", "fse", "se"]
    for col in TRANSFORMED_COLS + optional_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if EXCLUDE_INITIAL_DATA_FROM_PLOT and "f" in df.columns:
        df = df[(df["f"].isna()) | (df["f"] < 0.999999)].copy()

    return df


def plot_model_and_data(ax, series, central_df, mc_df, region, data_df,
                        x_col, y_col, xerr_col=None, yerr_col=None):
    color = style_value(SERIES_COLORS, series, None)
    region_color = style_value(SERIES_REGION_COLORS, series, color)
    linestyle = style_value(SERIES_LINESTYLES, series, "-")
    linewidth = style_value(SERIES_LINEWIDTHS, series, 2.5)
    marker = style_value(SERIES_MARKERS, series, "o")
    markersize = style_value(SERIES_MARKER_SIZES, series, 9)
    markerfacecolor = style_value(SERIES_MARKER_FACE_COLORS, series, color)
    markeredgecolor = style_value(SERIES_MARKER_EDGE_COLORS, series, color)
    markeredgewidth = style_value(SERIES_MARKER_EDGE_WIDTHS, series, 1.0)

    plot_uncertainty = SERIES_PLOT_UNCERTAINTY.get(series, True)

    if (plot_uncertainty and PLOT_INDIVIDUAL_MC_TRAJECTORIES
            and mc_df is not None and not mc_df.empty):
        for _sample_id, sub in mc_df.groupby("sample_id"):
            sub = sub.sort_values("f")
            ax.plot(
                sub[x_col],
                sub[y_col],
                color=color,
                alpha=MC_TRAJECTORY_ALPHA,
                linewidth=MC_TRAJECTORY_LINEWIDTH,
                label=None,
            )

    # Draw the wider 2-sigma region first, using a lighter color.
    if plot_uncertainty and "2sigma" in MC_INTERVALS_TO_PLOT:
        draw_parametric_region(
            ax,
            region[f"{x_col}_lo_2sigma"],
            region[f"{y_col}_lo_2sigma"],
            region[f"{x_col}_hi_2sigma"],
            region[f"{y_col}_hi_2sigma"],
            label=SERIES_REGION_LABELS_2SIGMA.get(
                series,
                f"{display_label(series)} ±2σ MC region",
            ),
            alpha=REGION_ALPHA_2SIGMA,
            color=region_color,
        )

    # Draw the narrower 1-sigma region on top, using a deeper color.
    if plot_uncertainty and "1sigma" in MC_INTERVALS_TO_PLOT:
        draw_parametric_region(
            ax,
            region[f"{x_col}_lo_1sigma"],
            region[f"{y_col}_lo_1sigma"],
            region[f"{x_col}_hi_1sigma"],
            region[f"{y_col}_hi_1sigma"],
            label=SERIES_REGION_LABELS_1SIGMA.get(
                series,
                f"{display_label(series)} ±1σ MC region",
            ),
            alpha=REGION_ALPHA_1SIGMA,
            color=region_color,
        )

    central_sorted = central_df.sort_values("f")
    ax.plot(
        central_sorted[x_col],
        central_sorted[y_col],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=SERIES_MODEL_LABELS.get(series, f"{display_label(series)} central model"),
    )

    if not data_df.empty:
        plot_df = data_df.dropna(subset=[x_col, y_col]).copy()
        if not plot_df.empty:
            xerr = plot_df[xerr_col] if xerr_col is not None and xerr_col in plot_df.columns else None
            yerr = plot_df[yerr_col] if yerr_col is not None and yerr_col in plot_df.columns else None
            ax.errorbar(
                plot_df[x_col],
                plot_df[y_col],
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                linestyle="none",
                color=color,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markeredgewidth=markeredgewidth,
                markersize=markersize,
                capsize=0,
                label=SERIES_DATA_LABELS_FOR_LEGEND.get(series, f"{display_label(series)} data"),
            )
        else:
            print(f"  no valid {x_col} / {y_col} experimental data for {series}; plotting model only.")
    else:
        print(f"  no experimental rows matched labels for {series}; plotting model only.")



# =============================================================================
# Additional f-axis plots
# =============================================================================

# Figure size for the four f-axis plots.
F_AXIS_FIGSIZE = (8, 4)

# Shared x-axis range and ticks for all four f-axis plots.
# Use reversed limits because f decreases during oxidation.
XLIM_F = (1.02, 0.0)
F_AXIS_X_MAJOR_INTERVAL = 0.2
F_AXIS_X_MINOR_INTERVAL = 0.04

# Only keep the legend on one of the four f-axis plots.
# Set to one of: "ln(c/c0)", "ln(d/d0)", "DeltaCD", "DeltaDD"
# Set to None to remove legends from all four f-axis plots.
F_AXIS_LEGEND_PLOT = "ln(c/c0)"

YLIM_F_PLOTS = {
    "ln(c/c0)": [-0.05,0.06],
    "ln(d/d0)": None,
    "DeltaCD": None,
    "DeltaDD": None,
}

# Output filename suffixes.
F_AXIS_PLOT_FILENAMES = {
    "ln(c/c0)": "fig_f_lnc_uncertainty.pdf",
    "ln(d/d0)": "fig_f_lnd_uncertainty.pdf",
    "DeltaCD": "fig_f_DeltaCD_uncertainty.pdf",
    "DeltaDD": "fig_f_DeltaDD_uncertainty.pdf",
}

# Axis labels.
F_AXIS_YLABELS = {
    "ln(c/c0)": r'ln$\frac{\delta^{13}{\rm C}+1000}{\delta^{13}{\rm C}_{\rm init}+1000}$',
    "ln(d/d0)": r'ln$\frac{\delta{\rm D}+1000}{\delta{\rm D}_{\rm init}+1000}$',
    "DeltaCD": r'$\Delta \Delta^{13}$CH$_3$D (‰)',
    "DeltaDD": r'$\Delta \Delta^{12}$CH$_2$D$_2$ (‰)',
}

# Experimental y-error columns.
# If a column is absent or all NaN, no y-error is plotted.
F_AXIS_YERR_COLS = {
    "ln(c/c0)": "lncse",
    "ln(d/d0)": "lndse",
    "DeltaCD": "cdse",
    "DeltaDD": "ddse",
}


def _valid_error_array(df, col):
    """
    Return numeric error array if available; otherwise return None.
    """
    if col is None or col not in df.columns:
        return None

    err = pd.to_numeric(df[col], errors="coerce")
    if err.notna().sum() == 0:
        return None

    return err


def plot_model_and_data_vs_f(ax, series, central_df, mc_df, region, data_df, y_col):
    """
    Plot one transformed quantity versus residual methane fraction f.

    x-axis:
        f

    y-axis:
        one of ln(c/c0), ln(d/d0), DeltaCD, DeltaDD
    """
    color = style_value(SERIES_COLORS, series, None)
    region_color = style_value(SERIES_REGION_COLORS, series, color)
    linestyle = style_value(SERIES_LINESTYLES, series, "-")
    linewidth = style_value(SERIES_LINEWIDTHS, series, 2.5)
    marker = style_value(SERIES_MARKERS, series, "o")
    markersize = style_value(SERIES_MARKER_SIZES, series, 9)
    markerfacecolor = style_value(SERIES_MARKER_FACE_COLORS, series, color)
    markeredgecolor = style_value(SERIES_MARKER_EDGE_COLORS, series, color)
    markeredgewidth = style_value(SERIES_MARKER_EDGE_WIDTHS, series, 1.0)

    # Optional individual Monte Carlo trajectories.
    plot_uncertainty = SERIES_PLOT_UNCERTAINTY.get(series, True)

    if (plot_uncertainty and PLOT_INDIVIDUAL_MC_TRAJECTORIES
            and mc_df is not None and not mc_df.empty):
        for _sample_id, sub in mc_df.groupby("sample_id"):
            sub = sub.sort_values("f")
            ax.plot(
                sub["f"],
                sub[y_col],
                color=color,
                alpha=MC_TRAJECTORY_ALPHA,
                linewidth=MC_TRAJECTORY_LINEWIDTH,
                label=None,
            )

    # 2-sigma region.
    if plot_uncertainty and "2sigma" in MC_INTERVALS_TO_PLOT:
        ax.fill_between(
            region["f"],
            region[f"{y_col}_lo_2sigma"],
            region[f"{y_col}_hi_2sigma"],
            color=region_color,
            alpha=REGION_ALPHA_2SIGMA,
            linewidth=0,
            label=SERIES_REGION_LABELS_2SIGMA.get(
                series,
                f"{display_label(series)} ±2σ MC region",
            ),
        )

    # 1-sigma region.
    if plot_uncertainty and "1sigma" in MC_INTERVALS_TO_PLOT:
        ax.fill_between(
            region["f"],
            region[f"{y_col}_lo_1sigma"],
            region[f"{y_col}_hi_1sigma"],
            color=region_color,
            alpha=REGION_ALPHA_1SIGMA,
            linewidth=0,
            label=SERIES_REGION_LABELS_1SIGMA.get(
                series,
                f"{display_label(series)} ±1σ MC region",
            ),
        )

    # Central deterministic model.
    central_sorted = central_df.sort_values("f")
    ax.plot(
        central_sorted["f"],
        central_sorted[y_col],
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        label=SERIES_MODEL_LABELS.get(series, f"{display_label(series)} central model"),
    )

    # Experimental data.
    if not data_df.empty:
        plot_df = data_df.dropna(subset=["f", y_col]).copy()

        if not plot_df.empty:
            xerr = _valid_error_array(plot_df, "fse")
            yerr = _valid_error_array(plot_df, F_AXIS_YERR_COLS.get(y_col))

            ax.errorbar(
                plot_df["f"],
                plot_df[y_col],
                xerr=xerr,
                yerr=yerr,
                fmt=marker,
                linestyle="none",
                color=color,
                markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor,
                markeredgewidth=markeredgewidth,
                markersize=markersize,
                capsize=0,
                label=SERIES_DATA_LABELS_FOR_LEGEND.get(
                    series,
                    f"{display_label(series)} data",
                ),
            )
        else:
            print(f"  no valid f / {y_col} experimental data for {series}; plotting model only.")
    else:
        print(f"  no experimental rows matched labels for {series}; plotting model only.")


def make_f_axis_plots(data, results_by_series):
    """
    Make four additional plots:

        ln(c/c0) vs f
        ln(d/d0) vs f
        DeltaCD vs f
        DeltaDD vs f

    User-requested formatting:
        1. rectangular plots, figsize=(12, 6)
        2. identical x-axis range and major/minor ticks for all four plots
        3. legend only on one selected plot
        4. y-axis minor ticks are 1/5 of the major-tick interval
    """
    for y_col in ["ln(c/c0)", "ln(d/d0)", "DeltaCD", "DeltaDD"]:
        fig, ax = plt.subplots(figsize=F_AXIS_FIGSIZE)

        for series in SERIES_TO_PLOT:
            if series not in results_by_series:
                continue

            central_df, mc_df, region = results_by_series[series]
            data_df = get_valid_data_for_series(data, series)

            plot_model_and_data_vs_f(
                ax=ax,
                series=series,
                central_df=central_df,
                mc_df=mc_df,
                region=region,
                data_df=data_df,
                y_col=y_col,
            )

        ax.set_xlabel(r"Residual methane fraction, $f$", fontdict=font_labels)
        ax.set_ylabel(F_AXIS_YLABELS[y_col], fontdict=font_labels)

        # Place the f-axis label and tick labels at the bottom, and place the
        # isotope-axis label and tick labels at the right. Tick marks remain
        # visible on all four sides.
        ax.xaxis.set_label_position("bottom")
        ax.yaxis.set_label_position("right")

        ax.tick_params(
            which="major",
            direction="out",
            top=True,
            bottom=True,
            left=True,
            right=True,
            length=8,
            width=2.5,
            labelsize=24,
        )
        ax.tick_params(
            which="minor",
            direction="out",
            top=True,
            bottom=True,
            left=True,
            right=True,
            length=4,
            width=2.0,
            labelsize=24,
        )
        ax.tick_params(
            axis="x",
            which="both",
            labelbottom=True,
            labeltop=False,
        )
        ax.tick_params(
            axis="y",
            which="both",
            labelleft=False,
            labelright=True,
        )

        # Same x-axis range and tick spacing for all four f-axis plots.
        if XLIM_F is not None:
            ax.set_xlim(XLIM_F)

        ax.xaxis.set_major_locator(MultipleLocator(F_AXIS_X_MAJOR_INTERVAL))
        ax.xaxis.set_minor_locator(MultipleLocator(F_AXIS_X_MINOR_INTERVAL))

        if YLIM_F_PLOTS.get(y_col) is not None:
            ax.set_ylim(YLIM_F_PLOTS[y_col])

        # Set y-axis minor ticks to 1/5 of the major-tick interval.
        # AutoMinorLocator(5) divides each major interval into 5 equal parts.
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))

        # Only keep the legend on one selected f-axis plot.
        if F_AXIS_LEGEND_PLOT is not None and y_col == F_AXIS_LEGEND_PLOT:
            add_legend_sorted_by_rev1(ax, fontsize=9)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

        fig.tight_layout()

        out_path = f"{OUT_PREFIX}_{F_AXIS_PLOT_FILENAMES[y_col]}"
        fig.savefig(out_path)
        print(f"  {out_path}")

def make_plots(data, results_by_series):
    fig_bulk0, ax_bulk0 = plt.subplots(figsize=(8, 8))

    for series in SERIES_TO_PLOT:
        if series not in results_by_series:
            continue
        central_df, mc_df, region = results_by_series[series]
        data_df = get_valid_data_for_series(data, series)
        plot_model_and_data(
            ax=ax_bulk0,
            series=series,
            central_df=central_df,
            mc_df=mc_df,
            region=region,
            data_df=data_df,
            x_col="ln(c/c0)",
            y_col="ln(d/d0)",
            xerr_col="lncse",
            yerr_col="lndse",
        )

    ax_bulk0.set_xlabel(
        r'ln$\frac{\delta^{13}{\rm C}+1000}{\delta^{13}{\rm C}_{\rm init}+1000}$',
        fontdict=font_labels,
    )
    ax_bulk0.set_ylabel(
        r'ln$\frac{\delta{\rm D}+1000}{\delta{\rm D}_{\rm init}+1000}$',
        fontdict=font_labels,
    )
    ax_bulk0.tick_params(which='major',direction='out', top=True, right=True, length=8, width=2.5, labelsize=24)
    ax_bulk0.tick_params(which='minor',direction='out', top=True, right=True, length=4, width=2.0, labelsize=24)
    ax_bulk0.xaxis.set_minor_locator(MultipleLocator(0.005))
    ax_bulk0.yaxis.set_minor_locator(MultipleLocator(0.01))
    if XLIM_BULK is not None:
        ax_bulk0.set_xlim(XLIM_BULK)
    if YLIM_BULK is not None:
        ax_bulk0.set_ylim(YLIM_BULK)
    add_legend_sorted_by_rev1(ax_bulk0, fontsize=10)
    fig_bulk0.tight_layout()

    fig_clump0, ax_clump0 = plt.subplots(figsize=(8, 8))

    for series in SERIES_TO_PLOT:
        if series not in results_by_series:
            continue
        central_df, mc_df, region = results_by_series[series]
        data_df = get_valid_data_for_series(data, series)
        plot_model_and_data(
            ax=ax_clump0,
            series=series,
            central_df=central_df,
            mc_df=mc_df,
            region=region,
            data_df=data_df,
            x_col="DeltaCD",
            y_col="DeltaDD",
        )

    ax_clump0.set_xlabel(r'$\Delta \Delta^{13}$CH$_3$D (‰)', fontdict=font_labels)
    ax_clump0.set_ylabel(r'$\Delta \Delta^{12}$CH$_2$D$_2$ (‰)', fontdict=font_labels)

    # Place the x-axis label and tick labels at the bottom, and place the y-axis
    # label and tick labels at the left. Tick marks remain visible on all sides.
    ax_clump0.xaxis.set_label_position("bottom")
    ax_clump0.yaxis.set_label_position("left")

    ax_clump0.tick_params(
        which="major",
        direction="out",
        top=True,
        bottom=True,
        left=True,
        right=True,
        length=8,
        width=2.5,
        labelsize=24,
    )
    ax_clump0.tick_params(
        which="minor",
        direction="out",
        top=True,
        bottom=True,
        left=True,
        right=True,
        length=4,
        width=2.0,
        labelsize=24,
    )
    ax_clump0.tick_params(
        axis="x",
        which="both",
        labeltop=False,
        labelbottom=True,
    )
    ax_clump0.tick_params(
        axis="y",
        which="both",
        labelright=False,
        labelleft=True,
    )
    ax_clump0.xaxis.set_minor_locator(MultipleLocator(2))
    ax_clump0.yaxis.set_minor_locator(MultipleLocator(4))
    if XLIM_CLUMP is not None:
        ax_clump0.set_xlim(XLIM_CLUMP)
    if YLIM_CLUMP is not None:
        ax_clump0.set_ylim(YLIM_CLUMP)
    # add_legend_sorted_by_rev1(ax_clump0, fontsize=10)
    fig_clump0.tight_layout()

    bulk_path = f"{OUT_PREFIX}_fig_bulk0_uncertainty.pdf"
    clump_path = f"{OUT_PREFIX}_fig_clump0_uncertainty.pdf"
    fig_bulk0.savefig(bulk_path)
    fig_clump0.savefig(clump_path)

    print("\nSaved figures:")
    print(f"  {bulk_path}")
    print(f"  {clump_path}")

    print("\nSaved f-axis figures:")
    make_f_axis_plots(data, results_by_series)


def main():
    total_start = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    model = load_model_prelude(MODEL_PY)
    validate_model_gamma_interface(model)
    data = pd.read_csv(DATA_CSV)
    specs = build_series_specs()

    results_by_series = {}
    settings_rows = []

    for series_name, spec in specs.items():
        central_df, mc_df, region, settings = run_monte_carlo_for_series(
            model=model,
            series_name=series_name,
            spec=spec,
            data=data,
            rng=rng,
        )
        results_by_series[series_name] = (central_df, mc_df, region)
        settings_rows.append(settings)

    settings_path = f"{OUT_PREFIX}_settings.csv"
    pd.DataFrame(settings_rows).to_csv(settings_path, index=False)
    print(f"\nSaved settings: {settings_path}")

    make_plots(data, results_by_series)

    print(f"Total runtime: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
