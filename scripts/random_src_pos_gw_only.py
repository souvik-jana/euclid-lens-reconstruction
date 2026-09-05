"""GW-only reconstructability as a function of where the source sits in the caustic.

One quad lens, ten source positions along two rays from the caustic centre -- one toward a
cusp, one toward a fold, five positions each, from near the centre out to just inside the
caustic. Five free parameters (T_star, dL, lens0_e2, y0gw, y1gw), two source-plane methods,
one summary table.

The companion sweep (euclid_lens_summary.py) varies the lens and holds the source at its
catalog position. This varies the source and holds the lens fixed.
"""

import os

# num_chains=8 needs eight host devices, or numpyro silently runs the chains sequentially.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import argparse
import copy
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_compilation_cache", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache" / "gwemfish-jax"))

import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
import pandas as pd
import scienceplots

plt.style.use(["science", "ieee", "high-vis"])
plt.rcParams["text.usetex"] = False

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from common import (
    EUCLID_VIS_BKG_RMS,
    EUCLID_VIS_FWHM,
    EUCLID_VIS_PIX_SCL,
    EUCLID_VIS_TEXP,
    NumpyEncoder,
    row_to_cfg,
)
from gwemfish import prune_gw_images
from gwemfish.corner_plot_utils import plot_multi_comparison_corner
from gwemfish.simple_pipeline import (
    _fisher_covariance,
    make_default_cfg,
    plot_posterior,
    plot_system_observation,
    run_inference,
    setup_em_observation,
    setup_gw_observation,
)
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.lens_model_extensions import LensModelExtensions

NPIX = 80
CATALOG = REPO_ROOT / "catalog" / "filtered_lens_catalog_PL_IC_gt_70.csv"
MODE = "GW-only"
METHODS = ("fisher-source", "deriv-approx-source")
TAGS = {"fisher-source": "fs", "deriv-approx-source": "das"}
COLORS = {"fisher-source": "steelblue", "deriv-approx-source": "darkorange"}
FREE_KEYS = ("T_star", "dL", "lens0_e2", "y0gw", "y1gw")
BOX_HALF_WIDTH = 0.5

# The caustic is resampled onto this many equally spaced angles before the cusps and folds
# are picked out; EXTREMUM_WINDOW is the half-window, in those samples, that a point has to
# beat to count as an extremum. 720/10 resolves the astroid's four lobes with room to spare.
ANGLE_SAMPLES = 720
EXTREMUM_WINDOW = 10

# Extra cusp-only points, as fractions of the cusp ray length. They start past the far end
# of the fold ray and walk in toward the cusp, covering the stretch no matched pair can.
EXTRA_CUSP_FRACTIONS = (0.6, 0.75, 0.9)

# Axis limits on the comparison corners, in sigma of the fisher-source result.
ZOOM_NSIGMA = 5.0

COLUMNS = (
    ["pos_id", "point", "ray", "frac", "y0_true", "y1_true", "r_from_centre", "r_caustic",
     "d_caustic", "box_half_width", "N_images", "min_img_sep", "max_img_sep",
     "mu_abs_min", "mu_abs_max"]
    + [f"{t}_{c}" for t in TAGS.values() for c in (
        "verdict", "sigma", "cond", "paramcount", "g0", "images", "observables",
        "srcbox", "caustic_margin", "cond_num", "g0_max", "max_pos_err",
        "sigma_T_star", "sigma_dL", "sigma_lens0_e2", "sigma_y0gw", "sigma_y1gw",
        "sigma_area", "seconds")]
    + ["verdict", "error"]
)


def build_cfg(row):
    """Catalog row -> GW-only config, with the Euclid VIS settings the sweep uses."""
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
    cfg["gw"]["source_box_half_width"] = BOX_HALF_WIDTH
    cfg["gw"]["error_scales"]["sigma_dL_eff"] = 0.005
    cfg["gw"]["error_scales"]["sigma_td"] = 0.001
    # setup_lens uses jaxtronomy whatever backend says, so jaxtronomy.solver governs the
    # truth positions; backend governs the finder inside the likelihood.
    cfg["gw"]["solver_params"]["backend"] = "jaxtronomy"
    # The closed-form EPL(+SHEAR) solver. It is the only configuration measured to be
    # self-consistent here: the "lenstronomy" grid solver finds more images at simulation
    # but then places them up to 4.6 arcsec from those same truth positions inside the
    # likelihood (images.ok False at every position), and that error is invariant to
    # min_distance and search_window, so it is structural rather than grid-limited.
    #
    # analytical has its own failure -- it silently drops the least-magnified image of a
    # quad on some geometries -- but that is a property of the lens, not a setting: it
    # moves with neither Nmeas nor magnification_limit. Lens 425 was picked by screening
    # for a system where this solver holds all four images at all 13 scan positions, so
    # N_images in the summary is the check that the choice still holds.
    cfg["gw"]["solver_params"]["jaxtronomy"]["solver"] = "analytical"
    cfg["use_parameter_layout"] = True
    cfg["inference"]["diagnostics"] = "warn"
    return row_to_cfg(row, cfg, True)


def priors_gw_only(tp, y0_lo, y0_hi, y1_lo, y1_hi):
    """The five free parameters; everything else pinned at truth.

    A quad gives 2*4-1 = 7 GW observables (3 time delays + 4 effective distances) against
    these 5 free parameters, so diagnostic check 4 passes with room to spare.
    """
    return {
        "T_star": dist.Uniform(1e1, 1e12),
        "dL": dist.Uniform(1e-5, 50000.0),
        "lens0_theta_E": float(tp["lens0_theta_E"]),
        "lens0_gamma": float(tp["lens0_gamma"]),
        "lens0_e1": float(tp["lens0_e1"]),
        "lens0_e2": dist.Uniform(-0.9, 0.9),
        "lens0_center_x": 0.0,
        "lens0_center_y": 0.0,
        "lens1_gamma1": float(tp["lens1_gamma1"]),
        "lens1_gamma2": float(tp["lens1_gamma2"]),
        "lens1_ra_0": 0.0,
        "lens1_dec_0": 0.0,
        "y0gw": dist.Uniform(y0_lo, y0_hi),
        "y1gw": dist.Uniform(y1_lo, y1_hi),
    }


def caustic_curves(kwargs_lens, lens_model_list, theta_E):
    """Tangential and radial caustics as (x, y) arrays.

    critical_curve_caustics returns a list of curves. In the source plane the tangential
    caustic is the astroid and the radial caustic is the larger oval around it, so the two
    are told apart by their maximum radius.
    """
    ext = LensModelExtensions(LensModel(lens_model_list=list(lens_model_list)))
    _, _, ra, dec = ext.critical_curve_caustics(
        kwargs_lens, compute_window=6 * theta_E, grid_scale=0.005
    )
    order = np.argsort([np.hypot(x, y).max() for x, y in zip(ra, dec)])
    inner, outer = order[0], order[-1]
    return (np.asarray(ra[inner]), np.asarray(dec[inner]),
            np.asarray(ra[outer]), np.asarray(dec[outer]))


def radial_extrema(radius, want_max):
    """Indices where the radius beats every other sample within +/-EXTREMUM_WINDOW.

    On an astroid the radial maxima are the four cusps and the radial minima are the
    midpoints of the four fold arcs. Strict comparison against the neighbours, rather than
    equality with the window extremum, so a flat run of equal radii yields one hit and not
    a whole plateau of them.
    """
    n = len(radius)
    beats = (np.greater if want_max else np.less)
    return [i for i in range(n)
            if all(beats(radius[i], radius[(i + j) % n])
                   for j in range(-EXTREMUM_WINDOW, EXTREMUM_WINDOW + 1) if j)]


def caustic_rays(x, y):
    """Centroid, and the (angle, radius) of every cusp and every fold midpoint.

    The raw curve is a traversal with uneven spacing and repeated points, which leaves ties
    that a window search reports as extra extrema. The astroid is star-shaped about its
    centroid, so radius is single-valued in angle: resampling onto a uniform angle grid
    removes the ties without changing the geometry.
    """
    cx, cy = float(x.mean()), float(y.mean())
    angle = np.arctan2(y - cy, x - cx)
    radius = np.hypot(x - cx, y - cy)
    order = np.argsort(angle)
    angle, radius = angle[order], radius[order]

    grid = np.linspace(-np.pi, np.pi, ANGLE_SAMPLES, endpoint=False)
    resampled = np.interp(grid, angle, radius, period=2 * np.pi)
    span = EXTREMUM_WINDOW * np.pi / ANGLE_SAMPLES
    cusps = [refine_extremum(angle, radius, grid[i], span, True)
             for i in radial_extrema(resampled, True)]
    folds = [refine_extremum(angle, radius, grid[i], span, False)
             for i in radial_extrema(resampled, False)]
    return (cx, cy), cusps, folds


def refine_extremum(angle, radius, centre_angle, span, want_max):
    """Exact (angle, radius) of the curve sample nearest an extremum found on the grid.

    A cusp is a corner, so linear interpolation across it cuts the tip -- measured 0.2277
    against a true 0.2338 on lens 1064. Locating on the grid and reading off the original
    samples keeps the resampling's robustness without its rounding.
    """
    offset = np.abs((angle - centre_angle + np.pi) % (2 * np.pi) - np.pi)
    window = np.flatnonzero(offset <= span)
    best = window[(np.argmax if want_max else np.argmin)(radius[window])]
    return float(angle[best]), float(radius[best])


def image_separations(ctx):
    """Closest and widest image pair, from the images the GW observables were built on."""
    x_img = np.asarray(ctx["x_img_gw"], dtype=float)
    y_img = np.asarray(ctx["y_img_gw"], dtype=float)
    sep = np.hypot(x_img[:, None] - x_img[None, :], y_img[:, None] - y_img[None, :])
    pairs = sep[np.triu_indices(len(x_img), 1)]
    return float(pairs.min()), float(pairs.max())


def image_magnifications(ctx):
    """|mu| at the fitted image positions, from ctx's own lens model.

    ctx carries the lens_gw object and the pruned image list that setup_gw_observation
    used to build dL_eff = dL/sqrt|mu|, so asking it is guaranteed consistent with the
    observables being fitted. Re-solving through a separate lenstronomy LensModel would be
    an independent calculation that can order or prune the images differently.
    """
    mu = ctx["lens_gw"].magnification(ctx["x_img_gw"], ctx["y_img_gw"], ctx["kwargs_lens"])
    return np.abs(np.asarray(mu, dtype=float))


def flag(ok):
    return "OK" if ok else "FAIL"


def tidy(value):
    """Trim a summary-table float to something readable.

    Three decimals everywhere would be wrong here: the columns run from sigma_area at
    ~1e-7 to the Fisher condition number at ~1e15, and fixed decimals write 0.0 for the
    first and eleven junk digits for the second. So three decimals in the range where that
    keeps at least three significant figures, and scientific notation outside it. The full
    precision is still in position.json and the pipeline JSON.
    """
    if not isinstance(value, float) or not np.isfinite(value):
        return value
    if value == 0.0:
        return 0.0
    if 0.01 <= abs(value) < 1e5:
        return round(value, 3)
    return f"{value:.3e}"


def fisher_covariance(ctx):
    keys = ctx["likelihood"]["keys_to_include"]
    H0 = np.asarray(ctx["fisher"]["H0"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(keys), np.asarray(_fisher_covariance(-H0, keys))


def sample_covariance(samples):
    keys = sorted(samples)
    stack = np.vstack([np.asarray(samples[k], dtype=float) for k in keys])
    return keys, np.cov(stack)


def localization_area(keys, cov):
    """sqrt(det) of the (y0gw, y1gw) block -- the sky-localization ellipse area proxy."""
    idx = [keys.index("y0gw"), keys.index("y1gw")]
    block = cov[np.ix_(idx, idx)]
    return float(np.sqrt(abs(np.linalg.det(block))))


def read_diagnostics(ctx):
    """The [diag] block as flags. Every check is optional.

    gwemfish drops a check entirely when it cannot be run -- ``observables`` disappears
    when the image check already failed badly enough that there is nothing to compare --
    so absence has to read as NA rather than raise.
    """
    diag = ctx["diagnostics"]
    cond = diag.get("conditioning", {})
    grad = diag.get("gradient", {})
    box = diag.get("source_box", {})
    images = diag.get("images")
    observables = diag.get("observables")
    return {
        "cond": flag(cond["ok"]) if cond else "NA",
        "cond_num": float(cond["condition_number"]) if cond else None,
        "g0": flag(grad["ok"]) if grad else "NA",
        "g0_max": float(grad["max_abs_scaled"]) if grad else None,
        "paramcount": flag(diag["parameters"]["ok"]) if "parameters" in diag else "NA",
        "images": flag(images["ok"]) if images else "NA",
        "observables": flag(observables["ok"]) if observables else "NA",
        # Check 3 is advisory, never fatal: a box past the caustic is a legitimate choice.
        "srcbox": "OK" if box.get("ok", True) else "WARN",
        "caustic_margin": box.get("caustic_margin"),
        "max_pos_err": (float(images["max_position_error"])
                        if images and images.get("max_position_error") is not None else None),
    }


def blank_method():
    record = dict.fromkeys(
        ("cond", "cond_num", "g0", "g0_max", "paramcount", "images", "observables",
         "srcbox", "caustic_margin", "seconds"), "ERROR"
    )
    record["verdict"] = "FAIL"
    record["sigma"] = "ERROR"
    for key in FREE_KEYS:
        record[f"sigma_{key}"] = None
    record["sigma_area"] = None
    return record


def run_method(ctx, method, pos_dir):
    """One inference run. Returns (record, samples, truths)."""
    t0 = time.time()
    inference = ({"informed": True, "regularize": False, "num_chains": 8,
                  "num_warmup": 8000, "num_samples": 12000}
                 if method == "deriv-approx-source" else {})
    samples, truths = run_inference(
        ctx, mode=MODE, method=method,
        cfg={
            "output": {
                "output_dir": pos_dir,
                "json_path": "pipeline.json",
                "save_samples_path": "samples.npz",
                "save_truths_path": "truths.npz",
            },
            "inference": inference,
        },
    )
    # fisher-source draws from inv(-H0) by construction, so its analytic covariance is the
    # exact answer and carries no sampling noise. deriv-approx-source samples a surrogate
    # that the uniform priors can truncate, so only its draws describe what came out.
    if method == "fisher-source":
        keys, cov = fisher_covariance(ctx)
    else:
        keys, cov = sample_covariance(samples)

    with np.errstate(invalid="ignore"):
        sigmas = np.sqrt(np.diag(cov))
    record = read_diagnostics(ctx)
    record["sigma"] = flag(not np.isnan(sigmas).any())
    # Finite sigmas alone are not enough. A solver that reproduces the wrong image
    # positions still returns a perfectly invertible Fisher matrix -- measured: positions
    # 4.6 arcsec off truth, every sigma finite, and the row read OK. The image and
    # observable checks are what catch that, so they gate the verdict too.
    record["verdict"] = flag(record["sigma"] == "OK"
                             and record["images"] == "OK"
                             and record["observables"] != "FAIL")
    for key in FREE_KEYS:
        record[f"sigma_{key}"] = float(sigmas[keys.index(key)])
    record["sigma_area"] = localization_area(keys, cov)
    record["seconds"] = round(time.time() - t0, 1)

    plot_posterior(
        samples, truths,
        cfg={
            "output": {"output_dir": os.path.join(pos_dir, method.replace("-", "_"))},
            "plot": {"plot_mode": "combined", "save_path": "source_plane_corner_all.png"},
        },
    )
    return record, samples, truths


def comparison_corner(results, pos_dir):
    """All free parameters in one overlay, axes zoomed to the fisher-source widths."""
    if len(results) < 2:
        print(f"[{pos_dir}] only one method ran; skipping the comparison corner")
        return
    labels = [m for m, _, _ in results]
    sample_sets = [s for _, s, _ in results]
    keys = sorted(set.intersection(*(set(s) for s in sample_sets)))

    truths = {}
    for _, _, t in results:
        truths.update(t)

    reference = next((s for m, s, _ in results if m == "fisher-source"), sample_sets[0])
    ranges = {}
    for key in keys:
        values = np.asarray(reference[key], dtype=float)
        mu, sd = float(values.mean()), float(values.std())
        if np.isfinite(sd) and sd > 0:
            lo, hi = mu - ZOOM_NSIGMA * sd, mu + ZOOM_NSIGMA * sd
            truth = truths.get(key)
            if truth is not None and np.isfinite(truth):
                lo, hi = min(lo, truth), max(hi, truth)
            ranges[key] = (lo, hi)

    plot_multi_comparison_corner(
        sample_sets,
        {"all": keys},
        labels=labels,
        colors=[COLORS[m] for m, _, _ in results],
        truths_dict={"all": {k: truths[k] for k in keys if k in truths}},
        param_ranges=ranges,
        save_path=os.path.join(pos_dir, "comparison_all_GW_only.png"),
        hist_kwargs={"density": True},
    )


def run_position(spec, cfg_base, out_root, methods):
    """Simulate and reconstruct one source position. Returns the summary row."""
    pos_dir = os.path.join(out_root, spec["label"])
    os.makedirs(pos_dir, exist_ok=True)
    row = dict.fromkeys(COLUMNS)
    row.update({k: spec[k] for k in
                ("pos_id", "point", "ray", "frac", "y0_true", "y1_true",
                 "r_from_centre", "r_caustic")})
    row["d_caustic"] = spec["d_caustic"]
    row["box_half_width"] = BOX_HALF_WIDTH
    errors = []

    cfg = copy.deepcopy(cfg_base)
    cfg["em"]["source_pos"] = (spec["y0_true"], spec["y1_true"])
    cfg["gw"]["source_pos"] = (spec["y0_true"], spec["y1_true"])

    # The EM observation is simulated but never fitted: GW-only inference ignores it, and
    # the EM priors are absent from priors_gw_only. It is here so plot_system_observation
    # has an image to draw -- that plotter needs ctx["lens_image"], which only the EM setup
    # builds. Setup is cheap on an 80x80 grid; it is EM *inference* that is not.
    ctx = setup_em_observation(cfg=cfg)
    ctx = setup_gw_observation(ctx, cfg=cfg)
    if len(ctx["x_img_gw"]) > 4:
        ctx = prune_gw_images(ctx, n_keep=4)
    ctx["cfg"]["gw"]["n_images"] = len(ctx["x_img_gw"])

    plot_system_observation(
        ctx,
        cfg={"output": {"output_dir": pos_dir,
                        "save_system_plot_path": "system_observation.png",
                        "system_plot_image_overlay": "gw"}},
    )

    mu = image_magnifications(ctx)
    row["N_images"] = len(ctx["x_img_gw"])
    row["min_img_sep"], row["max_img_sep"] = image_separations(ctx)
    row["mu_abs_min"], row["mu_abs_max"] = float(mu.min()), float(mu.max())

    y0_lo, y0_hi = spec["y0_true"] - BOX_HALF_WIDTH, spec["y0_true"] + BOX_HALF_WIDTH
    y1_lo, y1_hi = spec["y1_true"] - BOX_HALF_WIDTH, spec["y1_true"] + BOX_HALF_WIDTH
    ctx["cfg"]["priors"] = priors_gw_only(ctx["truth_params"], y0_lo, y0_hi, y1_lo, y1_hi)
    ctx["cfg"]["gw"]["source_plane_bounds"] = {"y0gw": (y0_lo, y0_hi), "y1gw": (y1_lo, y1_hi)}

    results = []
    for method in methods:
        tag = TAGS[method]
        try:
            # Each method owns its context: run_inference writes H0, diagnostics and the
            # likelihood block into it, and the second method must not read the first's.
            record, samples, truths = run_method(copy.deepcopy(ctx), method, pos_dir)
            results.append((method, samples, truths))
        except Exception as exc:
            record = blank_method()
            errors.append(f"{method}: {exc!r}")
        for key, value in record.items():
            row[f"{tag}_{key}"] = value

    comparison_corner(results, pos_dir)

    row["verdict"] = flag(all(row[f"{TAGS[m]}_verdict"] == "OK" for m in methods))
    row["error"] = " | ".join(errors)

    # position.json keeps the unrounded row; only the CSV is trimmed for reading.
    with open(os.path.join(pos_dir, "position.json"), "w") as fh:
        json.dump({"spec": spec, "row": row,
                   "x_img_gw": np.asarray(ctx["x_img_gw"]).tolist(),
                   "y_img_gw": np.asarray(ctx["y_img_gw"]).tolist()},
                  fh, indent=2, cls=NumpyEncoder)
    return {key: tidy(value) for key, value in row.items()}


def plot_caustic_scan(caustic_x, caustic_y, centre, cusps, folds, specs, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(caustic_x, caustic_y, "k-", lw=1.0, label="tangential caustic")
    for angle, radius in cusps:
        ax.plot(centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle),
                "^", color="0.4", ms=5)
    for angle, radius in folds:
        ax.plot(centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle),
                "s", color="0.4", ms=4)
    matched = {s["point"] for s in specs if s["ray"] == "fold"}
    for ray, colour in (("cusp", "crimson"), ("fold", "royalblue")):
        points = sorted((s for s in specs if s["ray"] == ray),
                        key=lambda s: s["r_from_centre"])
        ax.plot([centre[0], points[-1]["y0_true"]], [centre[1], points[-1]["y1_true"]],
                "-", color=colour, lw=0.7, alpha=0.6)
        # Filled = one of the matched pairs, open = a cusp-only extra beyond the fold ray.
        fill = ["full" if s["point"] in matched else "none" for s in points]
        for s, style in zip(points, fill):
            ax.plot(s["y0_true"], s["y1_true"], "o", ms=4, color=colour, zorder=3,
                    fillstyle=style)
            ax.annotate(s["point"], (s["y0_true"], s["y1_true"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=6, color=colour)
        ax.plot([], [], "o", ms=4, color=colour, label=f"{ray} ray")
    # Matched pairs sit on a shared circle: same distance from the centre, different
    # direction. Drawing them makes the apples-to-apples pairing visible.
    for radius in sorted({s["r_from_centre"] for s in specs if s["point"] in matched}):
        ax.add_patch(plt.Circle(centre, radius, fill=False, ec="0.75", lw=0.4, zorder=0))
    ax.plot(*centre, "k+", ms=8)
    ax.set_xlabel(r"$y_0$ [arcsec]")
    ax.set_ylabel(r"$y_1$ [arcsec]")
    ax.set_aspect("equal")
    ax.legend(fontsize=6)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_sigma_vs(table, path, xcolumn, xlabel, invert):
    """Every free parameter's width against one of the two distance measures.

    Both figures are produced because the two axes answer different questions and, on this
    lens, disagree: matched pairs share |y| but sit at very different caustic distances, so
    a vertical slice of the |y| figure is a clean cusp-vs-fold comparison at fixed source
    offset, while the d_caustic figure shows how proximity to the caustic itself acts.
    """
    quantities = [("sigma_y0gw", r"$\sigma(y_0)$ [arcsec]"),
                  ("sigma_y1gw", r"$\sigma(y_1)$ [arcsec]"),
                  ("sigma_area", r"$\sqrt{\det C_{yy}}$ [arcsec$^2$]"),
                  ("sigma_lens0_e2", r"$\sigma(e_2)$"),
                  ("sigma_dL", r"$\sigma(d_L)$ [Mpc]"),
                  ("sigma_T_star", r"$\sigma(T_\star)$ [s]")]
    fig, axes = plt.subplots(1, len(quantities), figsize=(3.0 * len(quantities), 2.8))
    for ax, (column, label) in zip(axes, quantities):
        for ray, style in (("cusp", "-"), ("fold", "--")):
            sub = table[table["ray"] == ray].sort_values(xcolumn)
            for method in METHODS:
                values = pd.to_numeric(sub[f"{TAGS[method]}_{column}"], errors="coerce")
                ax.plot(pd.to_numeric(sub[xcolumn], errors="coerce"), values,
                        marker="o" if ray == "cusp" else "s", ms=3, lw=0.8,
                        color=COLORS[method], linestyle=style,
                        label=f"{ray} / {method}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(label)
        ax.set_yscale("log")
        if invert:
            ax.invert_xaxis()
    axes[0].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--id", type=int, default=425,
                    help="catalog row index. 425 was chosen by screening every quad with "
                         "theta_E > 1 for one whose four images survive the analytical "
                         "solver at all 13 scan positions, then taking the best "
                         "caustic-size / detectability trade among those that passed.")
parser.add_argument("--fractions", default="0.1,0.3,0.5,0.7,0.9",
                    help="fractions of the FOLD ray length; these radii are used on both "
                         "rays, so each is a matched cusp/fold pair")
parser.add_argument("--methods", default=",".join(METHODS))
parser.add_argument("--positions", help="comma-separated position labels, e.g. A_cusp,A_fold")
parser.add_argument("--out", default=str(REPO_ROOT / "outputs" / "random-src-pos-analysis" / "GW-only"))
parser.add_argument("--resume", action="store_true", help="skip positions already in the CSV")
parser.add_argument("--dry-run", action="store_true", help="geometry and figures only, no inference")
args = parser.parse_args()

methods = [m.strip() for m in args.methods.split(",") if m.strip()]
unknown = [m for m in methods if m not in METHODS]
if unknown:
    parser.error(f"unknown method(s) {unknown}; choices are {list(METHODS)}")
fractions = [float(f) for f in args.fractions.split(",") if f.strip()]

out_root = Path(args.out)
out_root.mkdir(parents=True, exist_ok=True)
summary_path = out_root / "summary_gw_only.csv"

df = pd.read_csv(CATALOG)
row_catalog = df.iloc[args.id]
cfg_base = build_cfg(row_catalog)
kwargs_lens = cfg_base["lens"]["kwargs_lens"]
lens_model_list = cfg_base["lens"]["lens_model_list"]
theta_E = float(kwargs_lens[0]["theta_E"])

caustic_x, caustic_y, radial_x, radial_y = caustic_curves(kwargs_lens, lens_model_list, theta_E)
centre, cusps, folds = caustic_rays(caustic_x, caustic_y)
# The widest cusp and the narrowest fold: the two most distinct directions on the astroid.
cusp_ray = max(cusps, key=lambda c: c[1])
fold_ray = min(folds, key=lambda f: f[1])

print(f"lens {args.id}: theta_E = {theta_E:.4f}, gamma = {kwargs_lens[0]['gamma']:.4f}")
print(f"caustic centre ({centre[0]:+.5f}, {centre[1]:+.5f})")
print(f"cusps at r = {[round(c[1], 4) for c in cusps]}")
print(f"folds at r = {[round(f[1], 4) for f in folds]}")
print(f"cusp ray {np.degrees(cusp_ray[0]):+.2f} deg, r = {cusp_ray[1]:.4f}")
print(f"fold ray {np.degrees(fold_ray[0]):+.2f} deg, r = {fold_ray[1]:.4f}")

def make_spec(point, ray, angle, r_caustic, radius, pos_id):
    y0 = centre[0] + radius * np.cos(angle)
    y1 = centre[1] + radius * np.sin(angle)
    return {
        "pos_id": pos_id,
        "point": point,
        "label": f"{point}_{ray}",
        "ray": ray,
        "angle_deg": float(np.degrees(angle)),
        "r_caustic": float(r_caustic),
        "r_from_centre": float(radius),
        "frac": float(radius / r_caustic),
        "y0_true": float(y0),
        "y1_true": float(y1),
        "d_caustic": float(np.hypot(caustic_x - y0, caustic_y - y1).min()),
    }


# Matched pairs first: the same distance from the caustic centre on both rays, so the two
# directions differ only in direction. The fold ray is the shorter of the two (0.111 vs
# 0.234 on lens 1064), so the shared radii are set by it and every matched cusp point is
# comfortably inside its own caustic. The cusp-only extras then walk the rest of the way
# out, into the part of the cusp ray the fold ray cannot reach.
shared_radii = [f * fold_ray[1] for f in fractions]
extra_radii = [f * cusp_ray[1] for f in EXTRA_CUSP_FRACTIONS]

specs = []
for i, radius in enumerate(shared_radii):
    point = chr(ord("A") + i)
    specs.append(make_spec(point, "fold", fold_ray[0], fold_ray[1], radius, len(specs)))
    specs.append(make_spec(point, "cusp", cusp_ray[0], cusp_ray[1], radius, len(specs)))
for i, radius in enumerate(extra_radii):
    point = chr(ord("A") + len(shared_radii) + i)
    specs.append(make_spec(point, "cusp", cusp_ray[0], cusp_ray[1], radius, len(specs)))

with open(out_root / "scan_manifest.json", "w") as fh:
    json.dump({
        "catalog": CATALOG.name, "id": args.id,
        "kwargs_lens": kwargs_lens, "lens_model_list": lens_model_list,
        "zl": cfg_base["lens"]["zl"], "zs": cfg_base["lens"]["zs"],
        "npix": NPIX, "box_half_width": BOX_HALF_WIDTH,
        "free_keys": list(FREE_KEYS), "methods": methods, "fractions": fractions,
        "extra_cusp_fractions": list(EXTRA_CUSP_FRACTIONS),
        "shared_radii": shared_radii, "extra_cusp_radii": extra_radii,
        "caustic_centre": list(centre),
        "cusps": [{"angle_deg": float(np.degrees(a)), "r": r} for a, r in cusps],
        "folds": [{"angle_deg": float(np.degrees(a)), "r": r} for a, r in folds],
        "positions": specs,
    }, fh, indent=2, cls=NumpyEncoder)

np.savez_compressed(
    out_root / f"caustic_{args.id}.npz",
    tangential_x=caustic_x, tangential_y=caustic_y,
    radial_x=radial_x, radial_y=radial_y, centre=np.array(centre),
)
plot_caustic_scan(caustic_x, caustic_y, centre, cusps, folds, specs,
                  out_root / "caustic_scan.png")

print(f"\n{'label':10s}{'|y|':>9s}{'y0':>10s}{'y1':>10s}{'frac':>7s}{'d_caustic':>11s}")
for spec in sorted(specs, key=lambda s: (s["point"], s["ray"])):
    print(f"{spec['label']:10s}{spec['r_from_centre']:9.4f}{spec['y0_true']:10.4f}"
          f"{spec['y1_true']:10.4f}{spec['frac']:7.2f}{spec['d_caustic']:11.4f}")

selected = specs
if args.positions:
    wanted = [p.strip() for p in args.positions.split(",") if p.strip()]
    selected = [s for s in specs if s["label"] in wanted]
    missing = [w for w in wanted if w not in {s["label"] for s in specs}]
    if missing:
        parser.error(f"unknown position(s) {missing}; have {[s['label'] for s in specs]}")

done = set()
if args.resume and summary_path.exists():
    done = set(pd.read_csv(summary_path)["pos_id"].astype(int))
    selected = [s for s in selected if s["pos_id"] not in done]

if args.dry_run:
    print("\ndry run: geometry written, no inference")
    sys.exit(0)

print(f"\nrunning {len(selected)} position(s) x {len(methods)} method(s) -> {summary_path}")
write_header = not (args.resume and summary_path.exists())
t0 = time.time()
with open(summary_path, "w" if write_header else "a", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=COLUMNS)
    if write_header:
        writer.writeheader()
    for n, spec in enumerate(selected, 1):
        record = run_position(spec, cfg_base, str(out_root), methods)
        writer.writerow(record)
        fh.flush()
        print(f"[{n}/{len(selected)}] {spec['label']:12s} N={record['N_images']} "
              f"d_caustic={spec['d_caustic']:.4f} -> {record['verdict']} "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

table = pd.read_csv(summary_path)
plot_sigma_vs(table, out_root / "sigma_vs_offset_from_centre.png",
              "r_from_centre", r"$|y|$ from caustic centre [arcsec]", invert=False)
plot_sigma_vs(table, out_root / "sigma_vs_caustic_distance.png",
              "d_caustic", r"distance to caustic [arcsec]", invert=True)

print(f"\nwall clock {(time.time() - t0) / 60:.1f} min")
print(f"verdict: {(table['verdict'] == 'OK').sum()} OK / {(table['verdict'] == 'FAIL').sum()} FAIL")
print(table.sort_values(["point", "ray"])[
    ["point", "ray", "r_from_centre", "d_caustic", "N_images",
     "fs_verdict", "fs_srcbox", "fs_cond_num", "fs_sigma_y0gw", "fs_sigma_area",
     "das_verdict", "das_sigma_y0gw", "das_sigma_area"]].to_markdown(index=False,
                                                                    floatfmt=".4g"))
print(f"\nfull table: {summary_path}")
