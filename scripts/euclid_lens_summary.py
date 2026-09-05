"""Screen every Euclid-detectable catalog lens for Fisher reconstructability.

One cheap ``fisher`` run per mode (GW-only, EM-only, EM+GW) per lens. The verdict for a
mode is decided by the parameter sigmas alone: FAIL if any sigma comes back NaN. The
Fisher conditioning, GW parameter-count-vs-observable and g0 gradient checks are recorded
alongside it as context for reading a failure, but do not decide anything.

Writes one CSV. No plots, no pipeline JSON.
"""

import os

# One JAX device per process. Parallelism here is across worker processes, so the 20
# virtual CPU devices the other scripts request would be replicated in every worker --
# 3x20 device contexts and allocators, all idle.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import argparse
import contextlib
import copy
import csv
import gc
import io
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
# An explicit directory is what makes the compilation cache persistent. Without it the
# cache lives only in memory, and the per-lens jax.clear_caches() below would force a
# full recompile every time instead of a disk read. Shared by every worker.
jax.config.update("jax_compilation_cache_dir",
                  str(Path.home() / ".cache" / "gwemfish-jax"))

import numpy as np
import numpyro.distributions as dist
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import (
    EUCLID_VIS_BKG_RMS,
    EUCLID_VIS_FWHM,
    EUCLID_VIS_PIX_SCL,
    EUCLID_VIS_TEXP,
    row_to_cfg,
)
from gwemfish import prune_gw_images
from gwemfish.simple_pipeline import (
    _fisher_covariance,
    make_default_cfg,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)

NPIX = 80
CATALOG = REPO_ROOT / "catalog" / "filtered_lens_catalog_PL_IC_gt_70.csv"

COLUMNS = [
    "id", "IC_Euclid",
    "source_amp", "lens_light_amp", "theta_E", "lens_gamma", "sersic_n",
    "N_images",
    "em_sigma", "em_cond", "em_g0", "em_verdict",
    "gw_sigma", "gw_cond", "gw_paramcount", "gw_g0", "gw_verdict",
    "emgw_sigma", "emgw_cond", "emgw_g0", "emgw_verdict",
    "verdict",
    # Only the flags and the verdict are kept; a FAIL is investigated by re-running the
    # lens, which reprints the full [diag] block. ``error`` is the exception: when a mode
    # raises there is no [diag] block to reprint, so the message is the only record.
    "error",
]


def build_sample_cfg():
    cfg = make_default_cfg()
    cfg["em"]["pixel_grid_kwargs"] = {"npix": NPIX, "pix_scl": EUCLID_VIS_PIX_SCL}
    cfg["em"]["psf_kwargs"] = {"psf_type": "GAUSSIAN", "fwhm": EUCLID_VIS_FWHM}
    cfg["em"]["noise_simu_kwargs"] = {
        "npix": NPIX, "background_rms": EUCLID_VIS_BKG_RMS, "exposure_time": EUCLID_VIS_TEXP,
    }
    cfg["em"]["noise_inf_kwargs"] = {
        "npix": NPIX, "background_rms": None, "exposure_time": EUCLID_VIS_TEXP,
    }
    cfg["em"]["exposure_time"] = EUCLID_VIS_TEXP
    cfg["em"]["seed"] = 87651
    cfg["gw"]["image_box_half_width"] = 5.0
    cfg["gw"]["source_box_half_width"] = 0.03
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.1
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    # jaxtronomy's closed-form EPL(+SHEAR) solver returns the real images directly, with
    # no padding slot to mistake for a duplicate. n_images is deliberately left at the
    # default: the solver decides the count and _resolve_gw_n_images follows it.
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["use_parameter_layout"] = True
    # warn, not raise: a failing lens is a row in the table, not an abort.
    cfg["inference"]["diagnostics"] = "warn"
    return cfg


def priors_em_only(tp):
    return {
        "lens0_theta_E": dist.LogUniform(1e-3, 10.0),
        "lens0_gamma": float(tp["lens0_gamma"]),
        "lens0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
        "lens0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
        "lens0_center_x": 0.0,
        "lens0_center_y": 0.0,
        "lens1_gamma1": dist.Uniform(-0.3, 0.3),
        "lens1_gamma2": dist.Uniform(-0.3, 0.3),
        "lens1_ra_0": 0.0,
        "lens1_dec_0": 0.0,
        "source0_amp": dist.LogUniform(1e-6, 1e6),
        "source0_R_sersic": dist.Uniform(0.0, 30.0),
        "source0_n_sersic": dist.Uniform(0.05, 8.0),
        "source0_e1": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
        "source0_e2": dist.TruncatedNormal(0.0, 0.3, low=-1.0, high=1.0),
        "source0_center_x": dist.Normal(0.0, 0.3),
        "source0_center_y": dist.Normal(0.0, 0.3),
        "light0_amp": float(tp["light0_amp"]),
        "light0_R_sersic": float(tp["light0_R_sersic"]),
        "light0_n_sersic": float(tp["light0_n_sersic"]),
        "light0_e1": float(tp["light0_e1"]),
        "light0_e2": float(tp["light0_e2"]),
        "light0_center_x": float(tp["light0_center_x"]),
        "light0_center_y": float(tp["light0_center_y"]),
        "noise_sigma_bkg": float(tp["noise_sigma_bkg"]),
    }


def priors_gw_only(tp, n_img, y0_lo, y0_hi, y1_lo, y1_hi):
    """GW-only free set, scaled to what the image count can actually support.

    A GW-only system gives 2*N-1 numbers: N-1 time delays plus N effective distances.
    Free more than that and the Fisher matrix is degenerate by construction, so the
    free set grows with N:

        N=2 (3 obs)  ->  y0gw, y1gw                              (2 free)
        N=3 (5 obs)  ->  lens0_e2, y0gw, y1gw                    (3 free)
        N=4 (7 obs)  ->  T_star, dL, lens0_e2, y0gw, y1gw        (5 free)

    Anything not freed here is pinned at its truth value.
    """
    free_e2 = n_img >= 3
    free_dist = n_img >= 4
    return {
        # gwemfish's own default range for T_star (priors.py:19).
        "T_star": (dist.Uniform(1e1, 1e12) if free_dist else float(tp["T_star"])),
        "dL": (dist.Uniform(1e-5, 50000.0) if free_dist else float(tp["dL"])),
        "lens0_theta_E": float(tp["lens0_theta_E"]),
        "lens0_gamma": float(tp["lens0_gamma"]),
        "lens0_e1": float(tp["lens0_e1"]),
        "lens0_e2": (dist.Uniform(-0.9, 0.9) if free_e2 else float(tp["lens0_e2"])),
        "lens0_center_x": 0.0,
        "lens0_center_y": 0.0,
        "lens1_gamma1": float(tp["lens1_gamma1"]),
        "lens1_gamma2": float(tp["lens1_gamma2"]),
        "lens1_ra_0": 0.0,
        "lens1_dec_0": 0.0,
        "y0gw": dist.Uniform(y0_lo, y0_hi),
        "y1gw": dist.Uniform(y1_lo, y1_hi),
    }


def priors_em_gw(tp, y0_lo, y0_hi, y1_lo, y1_hi):
    priors = priors_em_only(tp)
    # priors["T_star"] = float(tp["T_star"])
    # priors["dL"] = float(tp["dL"])
    priors["y0gw"] = dist.Uniform(y0_lo, y0_hi)
    priors["y1gw"] = dist.Uniform(y1_lo, y1_hi)
    return priors


def flag(ok):
    return "OK" if ok else "FAIL"


def read_mode(ctx):
    """Pull the four checks out of a ctx that run_inference has just filled in."""
    keys = ctx["likelihood"]["keys_to_include"]
    H0 = np.asarray(ctx["fisher"]["H0"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov = np.asarray(_fisher_covariance(-H0, keys))
    with np.errstate(invalid="ignore"):
        sigmas = np.sqrt(np.diag(cov))
    n_nan = int(np.sum(np.isnan(sigmas)))

    diag = ctx["diagnostics"]
    cond = diag["conditioning"]
    grad = diag["gradient"]
    # check_parameter_count only runs when truth_params carries image_x* keys, so it is
    # absent for an EM-only context built before setup_gw_observation.
    par = diag.get("parameters")
    images = diag.get("images")
    observables = diag.get("observables")
    solver_ok = None
    if images is not None:
        solver_ok = bool(images["ok"])
        if observables is not None:
            solver_ok = solver_ok and bool(observables["ok"])

    return {
        "sigma": flag(n_nan == 0),
        "n_nan_sigma": n_nan,
        "cond": flag(cond["ok"]),
        "cond_num": float(cond["condition_number"]),
        "g0": flag(grad["ok"]),
        "g0_max": float(grad["max_abs_scaled"]),
        "paramcount": ("NA" if par is None or not par["applies"] else flag(par["ok"])),
        "n_free": (None if par is None else int(par["n_free"])),
        "n_obs": (None if par is None else par["n_gw_observables"]),
        "solver_ok": solver_ok,
        "verdict": flag(n_nan == 0),
    }


def blank_mode():
    return {
        "sigma": "ERROR", "n_nan_sigma": None, "cond": "ERROR", "cond_num": None,
        "g0": "ERROR", "g0_max": None, "paramcount": "ERROR", "n_free": None,
        "n_obs": None, "solver_ok": None, "verdict": "FAIL",
    }


def store(out, tag, result):
    """Fold a per-mode result into the row, dropping fields that mode has no column for
    (paramcount and n_obs are GW-only quantities, solver checks skip EM-only)."""
    for k, v in result.items():
        col = f"{tag}_{k}"
        if col in COLUMNS:
            out[col] = v


def run_mode(out, tag, mode, method, ctx, priors, src_bounds, errors):
    """Run one mode on a context it takes ownership of, and fold the result into the row."""
    ctx["cfg"]["priors"] = priors
    if src_bounds is not None:
        ctx["cfg"]["gw"]["source_plane_bounds"] = src_bounds
    try:
        run_inference(ctx, mode=mode, method=method)
        result = read_mode(ctx)
    except Exception as exc:
        result = blank_mode()
        errors.append(f"{mode}: {exc!r}")
    store(out, tag, result)


def gw_context(cfg, ctx=None):
    """Build a GW context and return it with its image count, pruned to at most 4.

    ``setup_gw_observation`` builds the lens geometry itself when handed an empty ctx,
    so GW-only needs nothing from the EM side.
    """
    ctx = setup_gw_observation({} if ctx is None else ctx, cfg=cfg)
    if len(ctx["x_img_gw"]) > 4:
        ctx = prune_gw_images(ctx, n_keep=4)
    n_img = len(ctx["x_img_gw"])
    ctx["cfg"]["gw"]["n_images"] = n_img
    return ctx, n_img


def source_plane_bounds(ctx):
    src = ctx["cfg"]["gw"]["source_pos"]
    hw = float(ctx["cfg"]["gw"]["source_box_half_width"])
    return (float(src[0]) - hw, float(src[0]) + hw,
            float(src[1]) - hw, float(src[1]) + hw)


def evaluate_lens(idx):
    """Screen one lens.

    The three modes are run independently: each builds its own context from ``cfg`` and
    carries its own try/except, so a failure in one never marks another. EM-only depends
    on the source and lens light; GW-only depends only on the mass model and the image
    positions; EM+GW needs both. Each verdict comes solely from that mode's own sigmas.
    """
    row = df.iloc[idx]
    out = dict.fromkeys(COLUMNS)
    out["id"] = int(idx)
    out["IC_Euclid"] = round(float(row["IC_Euclid"]), 3)
    out["theta_E"] = round(float(row["deflector_thetaE"]), 3)
    out["lens_gamma"] = round(float(row["deflector_slope"]), 3)
    out["sersic_n"] = round(float(row["source_sersic_index"]), 3)
    errors = []

    sink = io.StringIO()
    stream = contextlib.nullcontext() if VERBOSE else contextlib.redirect_stdout(sink)
    with stream, warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # cfg is the one genuinely shared input; nothing can run without it.
        try:
            cfg = row_to_cfg(row, SAMPLE_CFG, True)
            out["source_amp"] = round(float(cfg["em"]["kwargs_source"][0]["amp"]), 3)
            out["lens_light_amp"] = round(float(cfg["em"]["kwargs_lens_light"][0]["amp"]), 3)
        except Exception as exc:
            errors.append(f"cfg: {exc!r}")
            for tag in ("em", "gw", "emgw"):
                store(out, tag, blank_mode())
            out["verdict"] = "FAIL"
            out["error"] = " | ".join(errors)
            return out

        # --- EM-only: source + lens light, no GW. ---
        try:
            ctx = setup_em_observation(cfg=copy.deepcopy(cfg))
            run_mode(out, "em", "EM-only", "fisher", ctx,
                     priors_em_only(ctx["truth_params"]), None, errors)
            del ctx
        except Exception as exc:
            errors.append(f"EM-only setup: {exc!r}")
            store(out, "em", blank_mode())

        # --- GW-only: mass model + image positions, no light. ---
        try:
            ctx, n_img = gw_context(copy.deepcopy(cfg))
            out["N_images"] = n_img
            if n_img < 2:
                errors.append(f"GW-only: solver found {n_img} image(s); no GW observables")
                store(out, "gw", blank_mode())
            else:
                y0_lo, y0_hi, y1_lo, y1_hi = source_plane_bounds(ctx)
                run_mode(out, "gw", "GW-only", "fisher-source", ctx,
                         priors_gw_only(ctx["truth_params"], n_img,
                                        y0_lo, y0_hi, y1_lo, y1_hi),
                         {"y0gw": (y0_lo, y0_hi), "y1gw": (y1_lo, y1_hi)}, errors)
            del ctx
        except Exception as exc:
            errors.append(f"GW-only setup: {exc!r}")
            store(out, "gw", blank_mode())

        # --- EM+GW: needs both halves. ---
        try:
            cfg_both = copy.deepcopy(cfg)
            ctx, n_img = gw_context(cfg_both, ctx=setup_em_observation(cfg=cfg_both))
            if out["N_images"] is None:
                out["N_images"] = n_img
            if n_img < 2:
                errors.append(f"EM+GW: solver found {n_img} image(s); no GW observables")
                store(out, "emgw", blank_mode())
            else:
                y0_lo, y0_hi, y1_lo, y1_hi = source_plane_bounds(ctx)
                run_mode(out, "emgw", "EM+GW", "fisher-source", ctx,
                         priors_em_gw(ctx["truth_params"], y0_lo, y0_hi, y1_lo, y1_hi),
                         {"y0gw": (y0_lo, y0_hi), "y1gw": (y1_lo, y1_hi)}, errors)
            del ctx
        except Exception as exc:
            errors.append(f"EM+GW setup: {exc!r}")
            store(out, "emgw", blank_mode())

        # One lens peaks near 2.7 GB (the Fisher Hessian over the 80x80 EM image), then
        # each further lens adds only ~25 MB. Dropping the contexts and collecting covers
        # that drift; clearing the JAX compilation cache here as well was tried and cost
        # 3-4x throughput to recompile, for those same 25 MB.
        gc.collect()

    out["verdict"] = flag(
        all(out[f"{tag}_verdict"] == "OK" for tag in ("em", "gw", "emgw"))
    )
    out["error"] = " | ".join(errors)
    return out


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--start", type=int, default=0, help="offset into the IC>=70 list")
parser.add_argument("--limit", type=int, default=None, help="how many lenses to run")
parser.add_argument("--jobs", type=int, default=3, help="worker processes")
parser.add_argument("--recycle", type=int, default=25,
                    help="restart each worker after this many lenses, to cap JAX cache growth")
parser.add_argument("--ic-min", type=float, default=70.0, help="IC_Euclid threshold")
parser.add_argument("--ids", help="comma-separated catalog row ids; overrides the IC selection")
parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "euclid_lens_summary.csv"))
parser.add_argument("--resume", action="store_true", help="skip ids already in --out")
parser.add_argument("--verbose", action="store_true", help="let gwemfish [diag] output through")
args = parser.parse_args()

VERBOSE = args.verbose
SAMPLE_CFG = build_sample_cfg()
df = pd.read_csv(CATALOG)

# Guarded because the pool uses the "spawn" start method: a spawned child re-executes
# this module, and without the guard each child would build its own Pool forever.
if __name__ == "__main__":
    ids = list(df.index[df["IC_Euclid"] >= args.ic_min])
    if args.ids:
        selected = [int(t) for t in args.ids.split(",")]
    else:
        selected = ids[args.start:] if args.limit is None else ids[args.start:args.start + args.limit]

    out_path = Path(args.out)
    done = set()
    if args.resume and out_path.exists():
        done = set(pd.read_csv(out_path)["id"].astype(int))
        selected = [i for i in selected if i not in done]

    print(f"catalog {CATALOG.name}: {len(df)} rows, {len(ids)} with IC_Euclid >= {args.ic_min}")
    print(f"running {len(selected)} lenses on {args.jobs} worker(s) -> {out_path}")
    if done:
        print(f"resuming: {len(done)} already done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.resume and out_path.exists())
    mode = "w" if write_header else "a"

    t0 = time.time()
    n_done = 0
    with open(out_path, mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        ctxmp = mp.get_context("spawn")
        # Recycle workers: JAX caches a compiled executable per array shape, and the
        # image count varies from lens to lens, so a long-lived worker grows without
        # bound (measured: 1.7 GB after ~130 lenses, enough to push this machine into
        # swap and cost a 6x slowdown). Restarting costs one JIT warm-up per batch.
        with ctxmp.Pool(args.jobs, maxtasksperchild=args.recycle) as pool:
            for rec in pool.imap_unordered(evaluate_lens, selected):
                writer.writerow(rec)
                fh.flush()
                n_done += 1
                rate = (time.time() - t0) / n_done
                eta = rate * (len(selected) - n_done) / 60.0
                print(
                    f"[{n_done}/{len(selected)}] id={rec['id']:5d} "
                    f"N={rec['N_images']} "
                    f"gw={rec['gw_verdict']} em={rec['em_verdict']} emgw={rec['emgw_verdict']} "
                    f"-> {rec['verdict']}  ({rate:.1f}s/lens, ETA {eta:.0f} min)",
                    flush=True,
                )

    table = pd.read_csv(out_path)
    print(f"\nwall clock {(time.time() - t0) / 60:.1f} min for {n_done} lenses")
    print(f"\n{'=' * 70}\nverdict: {(table['verdict'] == 'OK').sum()} OK / "
          f"{(table['verdict'] == 'FAIL').sum()} FAIL of {len(table)}")
    for tag, label in (("gw", "GW-only"), ("em", "EM-only"), ("emgw", "EM+GW")):
        v = table[f"{tag}_verdict"]
        print(f"  {label:9s} {(v == 'OK').sum():5d} OK  {(v == 'FAIL').sum():5d} FAIL")
    print("\ncontext checks (not part of the verdict):")
    for col in ("gw_cond", "em_cond", "emgw_cond", "gw_paramcount",
                "gw_g0", "em_g0", "emgw_g0"):
        print(f"  {col:15s} {(table[col] == 'FAIL').sum():5d} FAIL")
    print(f"  N_images       {dict(table['N_images'].value_counts().sort_index())}")

    head = table[[
        "id", "theta_E", "lens_gamma", "sersic_n", "N_images",
        "em_sigma", "em_cond", "em_g0",
        "gw_sigma", "gw_cond", "gw_paramcount", "gw_g0",
        "emgw_sigma", "emgw_cond", "emgw_g0", "verdict",
    ]].head(30)
    print(f"\n{head.to_markdown(index=False, floatfmt='.3f')}")
    print(f"\nfull table: {out_path}")
