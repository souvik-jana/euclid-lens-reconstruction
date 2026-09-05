import copy
import json

import numpy as np
from lenstronomy.Util import param_util
from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from gwemfish.config import DEFAULT_KWARGS_NUMERICS, SOLVER_PARAMS

from lenstronomy.SimulationAPI.observation_api import SingleBand

_euclid_vis_cfg = Euclid("VIS", "GAUSSIAN").kwargs_single_band()
EUCLID_VIS_ZP      = _euclid_vis_cfg["magnitude_zero_point"]   # 25.72
EUCLID_VIS_PIX_SCL = _euclid_vis_cfg["pixel_scale"]            # 0.101 arcsec/px
EUCLID_VIS_FWHM    = _euclid_vis_cfg["seeing"]                 # 0.16 arcsec
EUCLID_VIS_TEXP    = _euclid_vis_cfg["exposure_time"]          # 566.0 s
EUCLID_VIS_BKG_RMS = SingleBand(**_euclid_vis_cfg).background_noise  # 0.011 e-/s/px


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.number):
            return obj.item()
        return super().default(obj)


sample_cfg = {
    "em": {
        "pixel_grid_kwargs": {"npix": int(40), "pix_scl": EUCLID_VIS_PIX_SCL},
        "psf_kwargs": {
            "psf_type": "GAUSSIAN",
            "fwhm": EUCLID_VIS_FWHM,
            "pixel_size": EUCLID_VIS_PIX_SCL,
        },
        "noise_simu_kwargs": {"npix": int(40), "background_rms": EUCLID_VIS_BKG_RMS, "exposure_time": EUCLID_VIS_TEXP},
        "noise_inf_kwargs":  {"npix": int(40), "background_rms": None,               "exposure_time": EUCLID_VIS_TEXP},
        "kwargs_numerics": DEFAULT_KWARGS_NUMERICS,
        "exposure_time": EUCLID_VIS_TEXP,
        "seed": 87651,
        "source_pos": (0.0, 0.0),
        "kwargs_source": [
            {
                "amp": 250,
                "R_sersic": 0.4,
                "n_sersic": 4.0,
                "e1": 0,
                "e2": 0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
        "kwargs_lens_light": [
            {
                "amp": 50.0,
                "R_sersic": 2.0,
                "n_sersic": 4.0,
                "e1": 0.0,
                "e2": 0.0,
                "center_x": 0.0,
                "center_y": 0.0,
            }
        ],
    },
    "lens": {
        "lens_model_list": ["EPL", "SHEAR"],
        "kwargs_lens": [
            {
                "theta_E": 1.2,
                "e1": 0.0,
                "e2": 0.1,
                "gamma": 2.0,
                "center_x": 0.0,
                "center_y": 0.0,
            },
            {"gamma1": 0.1, "gamma2": 0.0, "ra_0": 0.0, "dec_0": 0.0},
        ],
        "zl": 0.7,
        "zs": 1.5,
    },
    "gw": {
        "enabled": True,
        "n_images": 2,
        "source_pos": (0.0, 0.0),
        "solver_params": SOLVER_PARAMS,
        "image_box_half_width": 10.6,
        "error_scales": {
            "sigma_td": 0.05,
            "sigma_dL_eff": 0.2,
            "epsilon": 0.005,
        },
    },
    "plot": {"plot_mode": "groupwise", "save_path": None, "save_tag": None, "hist_kwargs": {"density": True}},
    "source_plane": {"filter_std": None, "use_filtered": False},
    "output": {
        "output_dir": "figures",
        "save_samples_path": None,
        "save_truths_path": None,
        "save_source_samples_path": None,
        "save_system_plot_path": None,
        "json_path": None,
    },
}


def row_to_cfg(row, cfg_template, gw_enabled):
    """
    Convert one row of the Qiuhan sky catalog into a GWEMFISH configuration.

    Parameters
    ----------
    row : pandas.Series
        One row of the lens catalog.
    cfg_template : dict
        Reference GWEMFISH config (e.g. sample_cfg).
    gw_enabled : bool
        Whether to include GW observation config.

    Returns
    -------
    cfg : dict
        GWEMFISH configuration for this lens system.
    """
    cfg = copy.deepcopy(cfg_template)

    lens_e1, lens_e2 = param_util.phi_q2_ellipticity(
        phi=row["deflector_pa"], q=row["deflector_q"]
    )
    source_e1, source_e2 = param_util.phi_q2_ellipticity(
        phi=row["source_pa"], q=row["source_q"]
    )

    kwargs_model = {
        "lens_light_model_list": ["SERSIC_ELLIPSE"],
        "source_light_model_list": ["SERSIC_ELLIPSE"],
    }
    mag_converter = MagAmpConversion(
        kwargs_model=kwargs_model, magnitude_zero_point=EUCLID_VIS_ZP
    )

    kwargs_lens_light_mag = [
        {
            "magnitude": row["deflector_app_mag_VIS"],
            "R_sersic": row["deflector_Re"],
            "n_sersic": 4.0,
            "e1": lens_e1,
            "e2": lens_e2,
            "center_x": 0,
            "center_y": 0,
        }
    ]
    kwargs_source_mag = [
        {
            "magnitude": row["source_app_mag_VIS"],
            "R_sersic": row["source_Re"],
            "n_sersic": row["source_sersic_index"],
            "e1": source_e1,
            "e2": source_e2,
            "center_x": row["source_relative_x"],
            "center_y": row["source_relative_y"],
        }
    ]

    lens_light_amp, source_amp, _ = mag_converter.magnitude2amplitude(
        kwargs_lens_light_mag=kwargs_lens_light_mag,
        kwargs_source_mag=kwargs_source_mag,
    )

    source_pos = (float(row["source_relative_x"]), float(row["source_relative_y"]))

    cfg["lens"]["zl"] = float(row["deflector_z"])
    cfg["lens"]["zs"] = float(row["source_z"])

    cfg["lens"]["kwargs_lens"][0]["theta_E"] = float(row["deflector_thetaE"])
    cfg["lens"]["kwargs_lens"][0]["gamma"]   = float(row["deflector_slope"])
    cfg["lens"]["kwargs_lens"][0]["e1"]      = float(lens_e1)
    cfg["lens"]["kwargs_lens"][0]["e2"]      = float(lens_e2)
    cfg["lens"]["kwargs_lens"][0]["center_x"] = 0.0
    cfg["lens"]["kwargs_lens"][0]["center_y"] = 0.0

    cfg["lens"]["kwargs_lens"][1]["gamma1"]  = float(row["deflector_shear1"])
    cfg["lens"]["kwargs_lens"][1]["gamma2"]  = float(row["deflector_shear2"])
    cfg["lens"]["kwargs_lens"][1]["ra_0"]    = 0.0
    cfg["lens"]["kwargs_lens"][1]["dec_0"]   = 0.0

    cfg["em"]["kwargs_source"][0]["R_sersic"] = float(row["source_Re"])
    cfg["em"]["kwargs_source"][0]["n_sersic"] = float(row["source_sersic_index"])
    cfg["em"]["kwargs_source"][0]["e1"]       = float(source_e1)
    cfg["em"]["kwargs_source"][0]["e2"]       = float(source_e2)
    cfg["em"]["kwargs_source"][0]["center_x"] = source_pos[0]
    cfg["em"]["kwargs_source"][0]["center_y"] = source_pos[1]
    cfg["em"]["kwargs_source"][0]["amp"]      = float(source_amp[0]["amp"])

    cfg["em"]["kwargs_lens_light"][0]["R_sersic"] = float(row["deflector_Re"])
    cfg["em"]["kwargs_lens_light"][0]["n_sersic"] = 4.0
    cfg["em"]["kwargs_lens_light"][0]["e1"]       = float(lens_e1)
    cfg["em"]["kwargs_lens_light"][0]["e2"]       = float(lens_e2)
    cfg["em"]["kwargs_lens_light"][0]["center_x"] = 0.0
    cfg["em"]["kwargs_lens_light"][0]["center_y"] = 0.0
    cfg["em"]["kwargs_lens_light"][0]["amp"]      = float(lens_light_amp[0]["amp"])

    cfg["em"]["source_pos"] = source_pos

    if gw_enabled:
        cfg["gw"]["source_pos"] = (source_pos[0] + 0.005, source_pos[1] - 0.005)
    else:
        cfg["gw"] = {"enabled": False}

    return cfg
