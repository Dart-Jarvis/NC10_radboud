#!/usr/bin/env python3
"""
Joint transformed + scale-normalized best-fit search for ANME2d + AOM_P.

This script fits the model to the already-normalized experimental columns:

    ln(c/c0), ln(d/d0), DeltaCD, DeltaDD

rather than fitting raw d13C, dD, D13CH3D, D12CH2D2.

Constraint:
    - ANME2d and AOM_P have separate rev1, rev2, rev3 values.
    - ANME2d and AOM_P share gammaCDff2 and gammaDDff2.

Default fitted parameters:
    ANME2d_rev1, ANME2d_rev2, ANME2d_rev3,
    AOM_P_rev1,  AOM_P_rev2,  AOM_P_rev3,
    gammaCDff2, gammaDDff2

Default bounds:
    rev1, rev2, rev3       : 0.00 to 0.99
    gammaCDff2             : 0.95 to 1.00
    gammaDDff2             : 0.90 to 1.00

Objective:
    - Exclude f = 1 rows by default.
    - For each series and each transformed variable, compute:
          residual = (model_transformed - observed_transformed) / observed_range
      where observed_range is max(obs) - min(obs) for that transformed variable
      within that series.

Usage:
    python joint_transformed_scaled_search.py \
        --model-py "multistep_model_final.py" \
        --data-csv "isotope_data.csv"

Fast test:
    python joint_transformed_scaled_search.py \
        --model-py "multistep_model_final.py" \
        --data-csv "isotope_data.csv" \
        --maxiter 20 --popsize 8 --ngrid 500

More serious run:
    python joint_transformed_scaled_search.py \
        --model-py "multistep_model_final.py" \
        --data-csv "isotope_data.csv" \
        --maxiter 120 --popsize 20 --ngrid 1000
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize


TARGET_COLS = ["ln(c/c0)", "ln(d/d0)", "DeltaCD", "DeltaDD"]


def load_model_prelude(model_py):
    """
    Load only the function/constant-definition part of the original model file.

    The original script runs expensive simulations and plotting at top level.
    This loader executes the file only up to the first large top-level simulation
    block, keeping constants, R0, dfdt, process(), and model_rev definitions.
    """
    model_py = Path(model_py)
    source = model_py.read_text()

    marker = "if len(rev1_list) != len(rev2_list)"
    if marker not in source:
        raise RuntimeError(
            "Could not find the top-level simulation marker. "
            "Check that this is the expected multistep_model_final.py file."
        )

    prelude = source.split(marker)[0]

    module = type(sys)("aom_model")
    module.__file__ = str(model_py)
    sys.modules["aom_model"] = module
    exec(compile(prelude, str(model_py), "exec"), module.__dict__)
    return module


def set_shared_gammas(model, gamma_cd, gamma_dd):
    """
    Update gammaCDff2/gammaDDff2 and their derived forward fractionation factors.

    In the model:
        a2cdff = gammaCDff2 * a2cff * a2dff
        a2ddff = gammaDDff2 * a2dff**2
    """
    model.gammaCDff2 = float(gamma_cd)
    model.gammaDDff2 = float(gamma_dd)
    model.a2cdff = model.gammaCDff2 * model.a2cff * model.a2dff
    model.a2ddff = model.gammaDDff2 * model.a2dff**2


def patch_model_solver(model, ngrid, rtol, atol, method):
    """
    Replace model.model_rev with a faster solve_ivp wrapper.

    Speed improvement:
        If f_stop is provided, the ODE integration stops as soon as the
        extracellular residual methane fraction fCH4 reaches f_stop, instead
        of always integrating to the safety cap tmax.

    It preserves the original model_rev return signature.
    """

    def fast_model_rev(rev1, rev2, rev3, tmax=None, num=None, f_stop=None):
        if tmax is None:
            tmax = 500.0
        if num is None:
            num = ngrid

        k1f_input = 1.0
        k1b_input = k1f_input * rev1
        k = [k1f_input, k1b_input]

        events = None
        if f_stop is not None:
            tot0 = float(np.sum(model.R0[12:17]))

            def reach_f_stop(_t, R, *unused_args):
                return float(np.sum(R[12:17]) / tot0 - f_stop)

            reach_f_stop.terminal = True
            reach_f_stop.direction = -1
            events = reach_f_stop

        sol = solve_ivp(
            model.dfdt,
            (model.t_lower, tmax),
            model.R0,
            args=(k, rev2, rev3),
            method=method,
            atol=atol,
            rtol=rtol,
            dense_output=(events is not None),
            events=events,
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        # If we used an event, evaluate the dense solution only up to the
        # event time. Otherwise evaluate the normal solution over [0, tmax].
        if events is not None and sol.t_events and len(sol.t_events[0]) > 0:
            t_end = float(sol.t_events[0][0])
        else:
            t_end = float(sol.t[-1])

        if events is not None:
            tint = np.linspace(model.t_lower, t_end, int(num))
            y_eval = sol.sol(tint)
            sol_eval = SimpleNamespace(t=tint, y=y_eval)
        else:
            tint = np.linspace(model.t_lower, t_end, int(num))
            y_eval = np.vstack([np.interp(tint, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])
            sol_eval = SimpleNamespace(t=tint, y=y_eval)

        (
            tot_sol,
            fCH4_sol,
            d13C_sol,
            dD_sol,
            D13CH3D_sol,
            D12CH2D2_sol,
        ) = model.process(sol_eval)

        # These are not used for fitting, but retain the original signature.
        rev1_sol = np.full_like(fCH4_sol, np.nan, dtype=float)
        rev2_sol = np.full_like(fCH4_sol, rev2, dtype=float)
        rev3_sol = np.full_like(fCH4_sol, rev3, dtype=float)

        return (
            tot_sol,
            fCH4_sol,
            rev1_sol,
            rev2_sol,
            rev3_sol,
            d13C_sol,
            dD_sol,
            D13CH3D_sol,
            D12CH2D2_sol,
        )

    model.model_rev = fast_model_rev


def model_transformed_trajectory(model, rev1, rev2, rev3, gamma_cd, gamma_dd,
                                 tmax, ngrid, f_stop=None):
    """
    Run the model and return transformed model outputs using the same
    transformations as normalize_dat() and normalize_clumped_dat():

        ln(c/c0)  = ln((d13C + 1000) / (d13C_initial + 1000))
        ln(d/d0)  = ln((dD   + 1000) / (dD_initial   + 1000))
        DeltaCD   = D13CH3D   - D13CH3D_initial
        DeltaDD   = D12CH2D2  - D12CH2D2_initial
    """
    set_shared_gammas(model, gamma_cd, gamma_dd)

    out = model.model_rev(rev1, rev2, rev3, tmax=tmax, num=ngrid, f_stop=f_stop)

    f_model = np.asarray(out[1], dtype=float)
    d13C = np.asarray(out[5], dtype=float)
    dD = np.asarray(out[6], dtype=float)
    D13CH3D = np.asarray(out[7], dtype=float)
    D12CH2D2 = np.asarray(out[8], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_c_c0 = np.log((d13C + 1000.0) / (d13C[0] + 1000.0))
        ln_d_d0 = np.log((dD + 1000.0) / (dD[0] + 1000.0))

    delta_cd = D13CH3D - D13CH3D[0]
    delta_dd = D12CH2D2 - D12CH2D2[0]

    return {
        "f": f_model,
        "ln(c/c0)": ln_c_c0,
        "ln(d/d0)": ln_d_d0,
        "DeltaCD": delta_cd,
        "DeltaDD": delta_dd,
    }


def interpolate_model_to_observed_f(model_curve, obs_f):
    """
    Interpolate transformed model outputs onto the observed f values.
    """
    f_model = np.asarray(model_curve["f"], dtype=float)

    # f decreases with time, so sort before interpolation.
    order = np.argsort(f_model)
    f_sorted = f_model[order]

    # Remove duplicate f values if any.
    f_unique, unique_idx = np.unique(f_sorted, return_index=True)

    pred = {}
    for col in TARGET_COLS:
        y_sorted = np.asarray(model_curve[col], dtype=float)[order][unique_idx]
        fn = interp1d(
            f_unique,
            y_sorted,
            kind="linear",
            bounds_error=False,
            fill_value=(y_sorted[0], y_sorted[-1]),
        )
        pred[col] = fn(obs_f)

    return pred


def prepare_series(data, label, exclude_initial=True):
    """
    Extract one series and compute observed ranges for scale normalization.
    """
    df = data[data["label"].astype(str) == str(label)].copy()

    if df.empty:
        raise ValueError(f"No data rows found for label={label!r}")

    df = df.dropna(subset=["f"] + TARGET_COLS)

    if exclude_initial:
        df = df[df["f"] < 0.999999].copy()

    if df.empty:
        raise ValueError(f"No usable rows remain for label={label!r}")

    df = df.sort_values("f", ascending=False).reset_index(drop=True)

    scales = {}
    for col in TARGET_COLS:
        scale = float(df[col].max() - df[col].min())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Non-positive scale for label={label!r}, column={col!r}")
        scales[col] = scale

    return df, scales


def residual_vector(theta, model, series_info, tmax, ngrid, f_margin, penalty=1e6):
    """
    Concatenate transformed, scale-normalized residuals for ANME2d and AOM_P.

    theta:
        [ANME2d_rev1, ANME2d_rev2, ANME2d_rev3,
         AOM_P_rev1,  AOM_P_rev2,  AOM_P_rev3,
         gammaCDff2, gammaDDff2]
    """
    theta = np.asarray(theta, dtype=float)

    anme_revs = theta[0:3]
    aomp_revs = theta[3:6]
    gamma_cd = theta[6]
    gamma_dd = theta[7]

    # Safety bounds. The optimizer also has bounds, but this protects direct calls.
    if np.any(anme_revs < 0.0) or np.any(anme_revs > 0.99):
        return np.full(10, penalty)
    if np.any(aomp_revs < 0.0) or np.any(aomp_revs > 0.99):
        return np.full(10, penalty)
    if not (0.95 <= gamma_cd <= 1.00):
        return np.full(10, penalty)
    if not (0.90 <= gamma_dd <= 1.00):
        return np.full(10, penalty)

    try:
        residuals = []

        for series_name, revs in [("ANME2d", anme_revs), ("AOM_P", aomp_revs)]:
            df, scales = series_info[series_name]
            obs_f = df["f"].to_numpy(dtype=float)

            f_stop = max(0.0, float(np.nanmin(obs_f)) - f_margin)

            curve = model_transformed_trajectory(
                model,
                rev1=revs[0],
                rev2=revs[1],
                rev3=revs[2],
                gamma_cd=gamma_cd,
                gamma_dd=gamma_dd,
                tmax=tmax,
                ngrid=ngrid,
                f_stop=f_stop,
            )

            pred = interpolate_model_to_observed_f(curve, obs_f)

            for col in TARGET_COLS:
                obs = df[col].to_numpy(dtype=float)
                res = (pred[col] - obs) / scales[col]
                residuals.extend(res)

        residuals = np.asarray(residuals, dtype=float)

        if not np.all(np.isfinite(residuals)):
            return np.full(10, penalty)

        return residuals

    except Exception:
        return np.full(10, penalty)


def objective(theta, model, series_info, tmax, ngrid, f_margin):
    r = residual_vector(theta, model, series_info, tmax, ngrid, f_margin)
    return float(np.sum(r**2))


def build_predictions_table(theta, model, series_info, tmax, ngrid, f_margin):
    """
    Create a table of observed values, model predictions, raw residuals,
    and scale-normalized residuals.
    """
    theta = np.asarray(theta, dtype=float)

    rows = []

    for series_name, revs in [("ANME2d", theta[0:3]), ("AOM_P", theta[3:6])]:
        df, scales = series_info[series_name]
        obs_f = df["f"].to_numpy(dtype=float)

        f_stop = max(0.0, float(np.nanmin(obs_f)) - f_margin)

        curve = model_transformed_trajectory(
            model,
            rev1=revs[0],
            rev2=revs[1],
            rev3=revs[2],
            gamma_cd=theta[6],
            gamma_dd=theta[7],
            tmax=tmax,
            ngrid=ngrid,
            f_stop=f_stop,
        )

        pred = interpolate_model_to_observed_f(curve, obs_f)

        for i, row in df.iterrows():
            out = {
                "series": series_name,
                "f": float(row["f"]),
                "rev1": float(revs[0]),
                "rev2": float(revs[1]),
                "rev3": float(revs[2]),
                "gammaCDff2": float(theta[6]),
                "gammaDDff2": float(theta[7]),
            }

            for col in TARGET_COLS:
                obs = float(row[col])
                mod = float(pred[col][i])
                out[f"{col}_obs"] = obs
                out[f"{col}_model"] = mod
                out[f"{col}_residual"] = mod - obs
                out[f"{col}_scaled_residual"] = (mod - obs) / scales[col]

            rows.append(out)

    return pd.DataFrame(rows)


def build_summary_table(predictions):
    """
    Summarize residuals by series and transformed isotope variable.
    """
    rows = []

    for series_name in predictions["series"].unique():
        sub = predictions[predictions["series"] == series_name]

        for col in TARGET_COLS:
            raw = sub[f"{col}_residual"].to_numpy(dtype=float)
            scaled = sub[f"{col}_scaled_residual"].to_numpy(dtype=float)

            rows.append({
                "series": series_name,
                "quantity": col,
                "n": len(raw),
                "raw_rmse": float(np.sqrt(np.mean(raw**2))),
                "scaled_rmse": float(np.sqrt(np.mean(scaled**2))),
                "scaled_ss": float(np.sum(scaled**2)),
            })

    all_scaled = []
    for col in TARGET_COLS:
        all_scaled.extend(predictions[f"{col}_scaled_residual"].to_numpy(dtype=float))

    all_scaled = np.asarray(all_scaled, dtype=float)

    rows.append({
        "series": "ALL",
        "quantity": "ALL",
        "n": len(all_scaled),
        "raw_rmse": np.nan,
        "scaled_rmse": float(np.sqrt(np.mean(all_scaled**2))),
        "scaled_ss": float(np.sum(all_scaled**2)),
    })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-py", required=True, help="Path to multistep_model_final.py")
    parser.add_argument("--data-csv", required=True, help="Path to isotope_data.csv")
    parser.add_argument("--out-prefix", default="joint_transformed_scaled")
    parser.add_argument("--include-initial", action="store_true", help="Include f=1 rows")
    parser.add_argument("--ngrid", type=int, default=900, help="ODE output grid size")
    parser.add_argument("--tmax", type=float, default=500.0, help="Safety cap for model integration time; event stop usually ends earlier")
    parser.add_argument("--f-margin", type=float, default=0.005, help="Integrate until model fCH4 reaches min(observed f) - this margin for each series")
    parser.add_argument("--rtol", type=float, default=1e-7)
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument("--method", default="LSODA", help="solve_ivp method: LSODA, BDF, RK45, etc.")
    parser.add_argument("--maxiter", type=int, default=90, help="Differential evolution iterations")
    parser.add_argument("--popsize", type=int, default=15, help="Differential evolution population size")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gammaDD-lower", type=float, default=0.90)
    parser.add_argument("--gammaCD-lower", type=float, default=0.95)
    args = parser.parse_args()

    model = load_model_prelude(args.model_py)
    patch_model_solver(
        model,
        ngrid=args.ngrid,
        rtol=args.rtol,
        atol=args.atol,
        method=args.method,
    )

    data = pd.read_csv(args.data_csv)

    series_info = {
        "ANME2d": prepare_series(data, label="3", exclude_initial=not args.include_initial),
        "AOM_P": prepare_series(data, label="AOM_P", exclude_initial=not args.include_initial),
    }

    print("\nPer-series ODE event stops:")
    for _name, (_df, _scales) in series_info.items():
        _f_min = float(np.nanmin(_df["f"].to_numpy(dtype=float)))
        _f_stop = max(0.0, _f_min - args.f_margin)
        print(f"  {_name}: min observed f = {_f_min:.6g}; integrate until f <= {_f_stop:.6g} or tmax cap")

    bounds = [
        (0.00, 0.99),  # ANME2d rev1
        (0.00, 0.99),  # ANME2d rev2
        (0.00, 0.99),  # ANME2d rev3
        (0.00, 0.99),  # AOM_P rev1
        (0.00, 0.99),  # AOM_P rev2
        (0.00, 0.99),  # AOM_P rev3
        (args.gammaCD_lower, 1.00),  # shared gammaCDff2
        (args.gammaDD_lower, 1.00),  # shared gammaDDff2
    ]

    print("Fitting transformed columns:")
    for col in TARGET_COLS:
        print(f"  {col}")
    print("\nBounds:")
    print(f"  gammaCDff2: {args.gammaCD_lower} to 1.00")
    print(f"  gammaDDff2: {args.gammaDD_lower} to 1.00")
    print(f"  exclude f=1 rows: {not args.include_initial}")

    print("\nStarting differential evolution...")
    de = differential_evolution(
        objective,
        bounds=bounds,
        args=(model, series_info, args.tmax, args.ngrid, args.f_margin),
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        polish=False,
        updating="immediate",
        workers=1,
        tol=1e-4,
        disp=True,
    )

    print("\nPolishing with L-BFGS-B...")
    local = minimize(
        objective,
        x0=de.x,
        args=(model, series_info, args.tmax, args.ngrid, args.f_margin),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-11},
    )

    if local.fun <= de.fun:
        best = local.x
        best_obj = float(local.fun)
        source = "differential_evolution + L-BFGS-B"
    else:
        best = de.x
        best_obj = float(de.fun)
        source = "differential_evolution"

    r = residual_vector(best, model, series_info, args.tmax, args.ngrid, args.f_margin)

    params = pd.DataFrame([{
        "ANME2d_rev1": best[0],
        "ANME2d_rev2": best[1],
        "ANME2d_rev3": best[2],
        "AOM_P_rev1": best[3],
        "AOM_P_rev2": best[4],
        "AOM_P_rev3": best[5],
        "gammaCDff2": best[6],
        "gammaDDff2": best[7],
        "scaled_ss": best_obj,
        "scaled_rmse": float(np.sqrt(np.mean(r**2))),
        "n_residuals": len(r),
        "n_parameters": len(best),
        "dof": len(r) - len(best),
        "reduced_scaled_ss": best_obj / max(len(r) - len(best), 1),
        "optimizer_source": source,
        "exclude_initial": not args.include_initial,
        "ngrid": args.ngrid,
        "tmax_safety_cap": args.tmax,
        "f_margin": args.f_margin,
    }])

    predictions = build_predictions_table(best, model, series_info, args.tmax, args.ngrid, args.f_margin)
    summary = build_summary_table(predictions)

    params_path = f"{args.out_prefix}_bestfit_params.csv"
    pred_path = f"{args.out_prefix}_predictions.csv"
    summary_path = f"{args.out_prefix}_summary.csv"

    params.to_csv(params_path, index=False)
    predictions.to_csv(pred_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\nBest-fit parameters:")
    print(params.T)

    print("\nResidual summary:")
    print(summary)

    print("\nWrote:")
    print(f"  {params_path}")
    print(f"  {pred_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
