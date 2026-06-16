#!/usr/bin/env python3
"""
FAST cache generator for fig_bulk0/fig_clump0 uncertainty plots.

This version fixes AOM_LS output:
    - It does NOT require AOM_LS experimental f / transformed columns to be present.
    - If AOM_LS experimental f is missing, it still generates the model region
      using a manually assigned f range.
    - You can tune the maximum ODE time globally or per series.

Run:
    python cache_fig0_uncertainty_trajectories_fast_v2.py

Then plot:
    python plot_fig0_uncertainty_from_cache_v2.py
"""

# =============================================================================
# USER SETTINGS
# =============================================================================

MODEL_PY = "multistep_model_final.py"
DATA_CSV = "isotope_data.csv"

# Keep this the same in the plotting script.
OUT_PREFIX = "fig0_uncertainty_cache"

INCLUDE_ANME2D = True
INCLUDE_AOM_P = True
INCLUDE_AOM_LS = True

ANME2D_LABEL = "3"
AOM_P_LABEL = "AOM_P"
AOM_LS_LABEL = "AOM_P_LS"

# Central reversibility values: [rev1, rev2, rev3]
ANME2D_REVS = [0.32, 0.97, 0.86]
AOM_P_REVS  = [0.78, 0.54, 0.03]
AOM_LS_REVS = [0.9, 0.96, 0.96]

# 1-sigma uncertainties for [rev1, rev2, rev3]
ANME2D_SIGMA_REVS = [0.04, 0.02, 0.04]
AOM_P_SIGMA_REVS  = [0.04, 0.04, 0.02]
AOM_LS_SIGMA_REVS = [0.03, 0.03, 0.03]

# Shared central gammas and uncertainties
GAMMA_CD_FF2 = 0.979
GAMMA_DD_FF2 = 0.92
SIGMA_GAMMA_CD_FF2 = 0.005
SIGMA_GAMMA_DD_FF2 = 0.005

# Bounds for finite-difference perturbations
REV_LOWER = 0.00
REV_UPPER = 0.99
GAMMA_CD_LOWER = 0.95
GAMMA_CD_UPPER = 1.00
GAMMA_DD_LOWER = 0.93
GAMMA_DD_UPPER = 1.00

# -------------------------------------------------------------------------
# Manual ODE time control
# -------------------------------------------------------------------------
# Global default maximum integration time.
DEFAULT_TMAX = 500.0

# Per-series maximum integration time. Set any value to override DEFAULT_TMAX.
# This is useful when a specific series requires longer/shorter model time.
SERIES_TMAX = {
    "ANME2d": 500.0,
    "AOM_P": 500.0,
    "AOM_LS": 500.0,
}

# If USE_EVENT_STOP=True, the solver stops when f reaches the target f_stop
# or when t reaches series TMAX, whichever comes first.
# If USE_EVENT_STOP=False, the solver always integrates to series TMAX.
USE_EVENT_STOP = True

# -------------------------------------------------------------------------
# Manual f-grid control
# -------------------------------------------------------------------------
# If a series has valid experimental f values, the script uses the minimum f
# in the data by default. If f is missing, it uses MANUAL_F_MIN[series].
#
# You can also force the manual f minimum even when data exist by setting
# USE_MANUAL_F_MIN[series] = True.
MANUAL_F_MIN = {
    "ANME2d": 0.3,
    "AOM_P": 0.1,
    "AOM_LS": 0.5,
}

USE_MANUAL_F_MIN = {
    "ANME2d": True,
    "AOM_P": True,
    "AOM_LS": True,   # AOM_LS often has missing f in the uploaded data.
}

# Stop slightly below f_min so interpolation covers the plotted curve.
F_MARGIN = 0.005

# Number of f points saved in each cached trajectory.
NGRID_F = 120

# ODE settings.
ODE_METHOD = "LSODA"
ODE_ATOL = 1e-9
ODE_RTOL = 1e-6

# If True, cache only selected series.
# Example: SERIES_TO_CACHE = ["AOM_LS"]
SERIES_TO_CACHE = None

# =============================================================================
# END USER SETTINGS
# =============================================================================

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


def load_model_prelude(model_py):
    model_py = Path(model_py)
    source = model_py.read_text()

    marker = "if len(rev1_list) != len(rev2_list)"
    if marker not in source:
        raise RuntimeError("Could not find the top-level simulation marker in the model file.")

    prelude = source.split(marker)[0]
    module = type(sys)("aom_model")
    module.__file__ = str(model_py)
    sys.modules["aom_model"] = module
    exec(compile(prelude, str(model_py), "exec"), module.__dict__)
    return module


def set_shared_gammas(model, gamma_cd, gamma_dd):
    model.gammaCDff2 = float(gamma_cd)
    model.gammaDDff2 = float(gamma_dd)
    model.a2cdff = model.gammaCDff2 * model.a2cff * model.a2dff
    model.a2ddff = model.gammaDDff2 * model.a2dff**2


def run_model_curve(model, revs, gamma_cd, gamma_dd, f_grid, f_stop, tmax):
    """
    Run one ODE trajectory and interpolate transformed outputs onto f_grid.
    """
    set_shared_gammas(model, gamma_cd, gamma_dd)

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


def add_run(rows, model, series_name, run_name, revs, gamma_cd, gamma_dd, f_grid,
            f_stop, tmax, central_revs, central_gammas):
    print(f"  running {series_name}: {run_name}")

    curve, elapsed, t_end, event_reached = run_model_curve(
        model, revs, gamma_cd, gamma_dd, f_grid, f_stop, tmax
    )

    status = "event reached" if event_reached else "tmax reached"
    print(f"    done in {elapsed:.2f} s; t_end={t_end:.3g}; {status}")

    curve.insert(0, "series", series_name)
    curve.insert(1, "run_name", run_name)
    curve.insert(2, "rev1", float(revs[0]))
    curve.insert(3, "rev2", float(revs[1]))
    curve.insert(4, "rev3", float(revs[2]))
    curve.insert(5, "gammaCDff2", float(gamma_cd))
    curve.insert(6, "gammaDDff2", float(gamma_dd))

    curve["central_rev1"] = float(central_revs[0])
    curve["central_rev2"] = float(central_revs[1])
    curve["central_rev3"] = float(central_revs[2])
    curve["central_gammaCDff2"] = float(central_gammas[0])
    curve["central_gammaDDff2"] = float(central_gammas[1])
    curve["solve_seconds"] = elapsed
    curve["t_end"] = t_end
    curve["tmax"] = tmax
    curve["event_reached"] = event_reached
    curve["f_stop"] = f_stop

    rows.append(curve)


def build_series_specs():
    specs = {}

    if INCLUDE_ANME2D:
        specs["ANME2d"] = {
            "label": ANME2D_LABEL,
            "revs": ANME2D_REVS,
            "rev_sigmas": ANME2D_SIGMA_REVS,
        }

    if INCLUDE_AOM_P:
        specs["AOM_P"] = {
            "label": AOM_P_LABEL,
            "revs": AOM_P_REVS,
            "rev_sigmas": AOM_P_SIGMA_REVS,
        }

    if INCLUDE_AOM_LS:
        specs["AOM_LS"] = {
            "label": AOM_LS_LABEL,
            "revs": AOM_LS_REVS,
            "rev_sigmas": AOM_LS_SIGMA_REVS,
        }

    if SERIES_TO_CACHE is not None:
        specs = {k: v for k, v in specs.items() if k in SERIES_TO_CACHE}

    if not specs:
        raise ValueError("No series selected.")

    return specs


def make_perturbations(revs, rev_sigmas):
    runs = []
    revs = np.asarray(revs, dtype=float)
    rev_sigmas = np.asarray(rev_sigmas, dtype=float)

    for i, name in enumerate(["rev1", "rev2", "rev3"]):
        sigma = float(rev_sigmas[i])
        if sigma <= 0:
            continue

        plus = revs.copy()
        minus = revs.copy()

        plus[i] = np.clip(plus[i] + sigma, REV_LOWER, REV_UPPER)
        minus[i] = np.clip(minus[i] - sigma, REV_LOWER, REV_UPPER)

        if plus[i] != revs[i]:
            runs.append((f"{name}_plus", plus.tolist(), GAMMA_CD_FF2, GAMMA_DD_FF2))
        if minus[i] != revs[i]:
            runs.append((f"{name}_minus", minus.tolist(), GAMMA_CD_FF2, GAMMA_DD_FF2))

    sigma_cd = float(SIGMA_GAMMA_CD_FF2)
    if sigma_cd > 0:
        gp = np.clip(GAMMA_CD_FF2 + sigma_cd, GAMMA_CD_LOWER, GAMMA_CD_UPPER)
        gm = np.clip(GAMMA_CD_FF2 - sigma_cd, GAMMA_CD_LOWER, GAMMA_CD_UPPER)
        if gp != GAMMA_CD_FF2:
            runs.append(("gammaCDff2_plus", revs.tolist(), gp, GAMMA_DD_FF2))
        if gm != GAMMA_CD_FF2:
            runs.append(("gammaCDff2_minus", revs.tolist(), gm, GAMMA_DD_FF2))

    sigma_dd = float(SIGMA_GAMMA_DD_FF2)
    if sigma_dd > 0:
        gp = np.clip(GAMMA_DD_FF2 + sigma_dd, GAMMA_DD_LOWER, GAMMA_DD_UPPER)
        gm = np.clip(GAMMA_DD_FF2 - sigma_dd, GAMMA_DD_LOWER, GAMMA_DD_UPPER)
        if gp != GAMMA_DD_FF2:
            runs.append(("gammaDDff2_plus", revs.tolist(), GAMMA_CD_FF2, gp))
        if gm != GAMMA_DD_FF2:
            runs.append(("gammaDDff2_minus", revs.tolist(), GAMMA_CD_FF2, gm))

    return runs


def determine_f_min(data, series_name, label):
    """
    Determine f_min for model cache.
    If data f is missing or USE_MANUAL_F_MIN is True, use MANUAL_F_MIN.
    """
    use_manual = USE_MANUAL_F_MIN.get(series_name, False)

    if not use_manual:
        df = data[data["label"].astype(str) == str(label)].copy()
        valid_f = pd.to_numeric(df["f"], errors="coerce").dropna()
        if len(valid_f) > 0:
            return float(valid_f.min()), "data"

    return float(MANUAL_F_MIN[series_name]), "manual"


def main():
    model = load_model_prelude(MODEL_PY)
    data = pd.read_csv(DATA_CSV)
    series_specs = build_series_specs()

    settings_rows = []
    total_start = time.time()

    for series_name, spec in series_specs.items():
        f_min, f_source = determine_f_min(data, series_name, spec["label"])
        f_stop = max(0.0, f_min - F_MARGIN)
        f_grid = np.linspace(f_min, 1.0, NGRID_F)
        tmax = float(SERIES_TMAX.get(series_name, DEFAULT_TMAX))

        rows = []
        central_gammas = [GAMMA_CD_FF2, GAMMA_DD_FF2]

        print(f"\nGenerating cached trajectories for {series_name}...")
        print(f"  label = {spec['label']}")
        print(f"  f_min = {f_min:.5f} ({f_source}); f_stop = {f_stop:.5f}")
        print(f"  tmax = {tmax}")
        print(f"  central revs = {spec['revs']}")
        print(f"  rev sigmas   = {spec['rev_sigmas']}")

        add_run(
            rows=rows,
            model=model,
            series_name=series_name,
            run_name="central",
            revs=spec["revs"],
            gamma_cd=GAMMA_CD_FF2,
            gamma_dd=GAMMA_DD_FF2,
            f_grid=f_grid,
            f_stop=f_stop,
            tmax=tmax,
            central_revs=spec["revs"],
            central_gammas=central_gammas,
        )

        for run_name, revs, gamma_cd, gamma_dd in make_perturbations(spec["revs"], spec["rev_sigmas"]):
            add_run(
                rows=rows,
                model=model,
                series_name=series_name,
                run_name=run_name,
                revs=revs,
                gamma_cd=gamma_cd,
                gamma_dd=gamma_dd,
                f_grid=f_grid,
                f_stop=f_stop,
                tmax=tmax,
                central_revs=spec["revs"],
                central_gammas=central_gammas,
            )

        out = pd.concat(rows, ignore_index=True)
        out_path = f"{OUT_PREFIX}_{series_name}_trajectories.csv"
        out.to_csv(out_path, index=False)
        print(f"  saved {out_path}")

        settings_rows.append({
            "series": series_name,
            "label": spec["label"],
            "rev1": spec["revs"][0],
            "rev2": spec["revs"][1],
            "rev3": spec["revs"][2],
            "sigma_rev1": spec["rev_sigmas"][0],
            "sigma_rev2": spec["rev_sigmas"][1],
            "sigma_rev3": spec["rev_sigmas"][2],
            "gammaCDff2": GAMMA_CD_FF2,
            "gammaDDff2": GAMMA_DD_FF2,
            "sigma_gammaCDff2": SIGMA_GAMMA_CD_FF2,
            "sigma_gammaDDff2": SIGMA_GAMMA_DD_FF2,
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
        })

    settings_path = f"{OUT_PREFIX}_settings.csv"
    pd.DataFrame(settings_rows).to_csv(settings_path, index=False)

    print(f"\nSaved settings: {settings_path}")
    print(f"Total cache generation time: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
