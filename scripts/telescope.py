import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.ObservationConfig.HST import HST
from lenstronomy.SimulationAPI.ObservationConfig.JWST import JWST
from lenstronomy.SimulationAPI.ObservationConfig.LSST import LSST
from lenstronomy.SimulationAPI.ObservationConfig.DES import DES
from lenstronomy.SimulationAPI.ObservationConfig.Roman import Roman
from lenstronomy.SimulationAPI.ObservationConfig.ZTF import ZTF
from lenstronomy.SimulationAPI.observation_api import SingleBand

BANDS = [
    ("Euclid", "VIS",          lambda: Euclid("VIS", "GAUSSIAN")),
    ("Euclid", "Y",            lambda: Euclid("Y",   "GAUSSIAN")),
    ("Euclid", "J",            lambda: Euclid("J",   "GAUSSIAN")),
    ("Euclid", "H",            lambda: Euclid("H",   "GAUSSIAN")),
    ("HST",    "WFC3_F160W",   lambda: HST("WFC3_F160W",  "GAUSSIAN")),
    ("HST",    "TDLMC_F160W",  lambda: HST("TDLMC_F160W", "PIXEL")),
    ("JWST",   "F115W",        lambda: JWST("F115W",  "PIXEL")),
    ("JWST",   "F150W",        lambda: JWST("F150W",  "PIXEL")),
    ("JWST",   "F200W",        lambda: JWST("F200W",  "PIXEL")),
    ("JWST",   "F277W",        lambda: JWST("F277W",  "PIXEL")),
    ("JWST",   "F356W",        lambda: JWST("F356W",  "PIXEL")),
    ("JWST",   "F444W",        lambda: JWST("F444W",  "PIXEL")),
    ("LSST",   "u",            lambda: LSST("u", "GAUSSIAN")),
    ("LSST",   "g",            lambda: LSST("g", "GAUSSIAN")),
    ("LSST",   "r",            lambda: LSST("r", "GAUSSIAN")),
    ("LSST",   "i",            lambda: LSST("i", "GAUSSIAN")),
    ("LSST",   "z",            lambda: LSST("z", "GAUSSIAN")),
    ("LSST",   "y",            lambda: LSST("y", "GAUSSIAN")),
    ("DES",    "g",            lambda: DES("g", "GAUSSIAN")),
    ("DES",    "r",            lambda: DES("r", "GAUSSIAN")),
    ("DES",    "i",            lambda: DES("i", "GAUSSIAN")),
    ("DES",    "z",            lambda: DES("z", "GAUSSIAN")),
    ("DES",    "Y",            lambda: DES("Y", "GAUSSIAN")),
    # Roman bands require survey_mode matched to the filter
    ("Roman",  "F062",         lambda: Roman("F062", "GAUSSIAN", survey_mode="time_domain_wide")),
    ("Roman",  "F087",         lambda: Roman("F087", "GAUSSIAN", survey_mode="time_domain_wide")),
    ("Roman",  "F106",         lambda: Roman("F106", "GAUSSIAN", survey_mode="wide_area")),
    ("Roman",  "F129",         lambda: Roman("F129", "GAUSSIAN", survey_mode="wide_area")),
    ("Roman",  "F158",         lambda: Roman("F158", "GAUSSIAN", survey_mode="wide_area")),
    ("Roman",  "F184",         lambda: Roman("F184", "GAUSSIAN", survey_mode="time_domain_deep")),
    ("Roman",  "F146",         lambda: Roman("F146", "GAUSSIAN", survey_mode="microlensing")),
    ("ZTF",    "g",            lambda: ZTF("g", "GAUSSIAN")),
    ("ZTF",    "r",            lambda: ZTF("r", "GAUSSIAN")),
    ("ZTF",    "i",            lambda: ZTF("i", "GAUSSIAN")),
]

header = (
    f"{'Telescope':<10} {'Band':<15} {'ZP':>6}  {'PSF':>9}  "
    f"{'FWHM\"':>6}  {'pix_scl\"':>9}  {'t_exp(s)':>9}  {'bkg_rms(e/s)':>13}  {'pixel_PSF':>10}"
)
separator = "-" * len(header)

print(header)
print(separator)

prev_tel = None
for tel, band, make in BANDS:
    if prev_tel is not None and tel != prev_tel:
        print()
    prev_tel = tel

    inst = make()
    cfg = inst.kwargs_single_band()

    zp    = cfg.get("magnitude_zero_point", "?")
    psf   = cfg.get("psf_type", "?")
    fwhm  = cfg.get("seeing")
    pix   = cfg.get("pixel_scale", "?")
    texp  = cfg.get("exposure_time")
    pixel_psf = psf == "PIXEL"

    fwhm_str = f"{fwhm:.3f}" if fwhm is not None else "N/A"
    texp_str = f"{texp:.1f}" if texp is not None else "N/A"

    band_obj = SingleBand(**cfg)
    bkg = band_obj.background_noise
    bkg_str = f"{bkg:.5f}"

    print(
        f"{tel:<10} {band:<15} {zp:>6.3f}  {psf:>9}  "
        f"{fwhm_str:>6}  {pix:>9.3f}  {texp_str:>9}  {bkg_str:>13}  {str(pixel_psf):>10}"
    )

print()
print("pixel_PSF = True: psf_type='PIXEL' is supported but the kernel map must be")
print("                  provided externally at runtime (not bundled in lenstronomy).")
