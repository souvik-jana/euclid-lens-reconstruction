# Telescope Specifications

All values sourced from `lenstronomy.SimulationAPI.ObservationConfig` (via `kwargs_single_band()`).  
`bkg_rms` computed via `SingleBand(**cfg).background_noise` = sqrt(read_noise² + sky_e⁻) / t_exp.  
Roman bands use the survey mode that supports each filter (see footnote).

| Telescope | Band         | ZP (AB) | PSF      | FWHM (") | pix_scl (") | t_exp (s) | bkg_rms (e⁻/s) | pixel_PSF |
|-----------|--------------|--------:|----------|----------:|------------:|----------:|---------------:|-----------|
| Euclid    | VIS          |  25.720 | GAUSSIAN |     0.160 |       0.101 |     566.0 |        0.01090 | —         |
| Euclid    | Y            |  25.040 | GAUSSIAN |     0.480 |       0.300 |      87.2 |        0.07137 | —         |
| Euclid    | J            |  25.260 | GAUSSIAN |     0.490 |       0.300 |      87.2 |        0.07447 | —         |
| Euclid    | H            |  25.210 | GAUSSIAN |     0.500 |       0.300 |      87.2 |        0.07062 | —         |
| HST       | WFC3_F160W   |  25.960 | GAUSSIAN |     0.080 |       0.080 |    5400.0 |        0.00592 | —         |
| HST       | TDLMC_F160W  |  25.946 | PIXEL    |       N/A |       0.080 |    5400.0 |        0.00674 | ✓ ²       |
| JWST      | F115W        |  28.020 | PIXEL    |     0.040 |       0.031 |     257.0 |        0.02170 | ✓ ²       |
| JWST      | F150W        |  28.020 | PIXEL    |     0.050 |       0.031 |     257.0 |        0.02170 | ✓ ²       |
| JWST      | F200W        |  28.000 | PIXEL    |       N/A |       0.031 |    3600.0 |        0.00439 | ✓ ²       |
| JWST      | F277W        |  26.490 | PIXEL    |     0.092 |       0.063 |     257.0 |        0.01823 | ✓ ²       |
| JWST      | F356W        |  26.470 | PIXEL    |       N/A |       0.063 |    3600.0 |        0.00371 | ✓ ²       |
| JWST      | F444W        |  26.490 | PIXEL    |     0.145 |       0.063 |     257.0 |        0.01824 | ✓ ²       |
| LSST      | u            |  26.500 | GAUSSIAN |     0.810 |       0.200 |      15.0 |        0.06048 | —         |
| LSST      | g            |  28.300 | GAUSSIAN |     0.770 |       0.200 |      15.0 |        0.07548 | —         |
| LSST      | r            |  28.130 | GAUSSIAN |     0.730 |       0.200 |      15.0 |        0.06630 | —         |
| LSST      | i            |  27.790 | GAUSSIAN |     0.710 |       0.200 |      15.0 |        0.07637 | —         |
| LSST      | z            |  27.400 | GAUSSIAN |     0.690 |       0.200 |      15.0 |        0.09950 | —         |
| LSST      | y            |  26.580 | GAUSSIAN |     0.680 |       0.200 |      15.0 |        0.10672 | —         |
| DES       | g            |  26.580 | GAUSSIAN |     1.120 |       0.263 |      90.0 |        0.07601 | —         |
| DES       | r            |  26.780 | GAUSSIAN |     0.960 |       0.263 |      90.0 |        0.11973 | —         |
| DES       | i            |  26.750 | GAUSSIAN |     0.880 |       0.263 |      90.0 |        0.20792 | —         |
| DES       | z            |  26.480 | GAUSSIAN |     0.840 |       0.263 |      90.0 |        0.31346 | —         |
| DES       | Y            |  25.400 | GAUSSIAN |     0.900 |       0.263 |      45.0 |        0.38453 | —         |
| Roman     | F062 ¹       |  26.618 | GAUSSIAN |     0.058 |       0.110 |      60.0 |        0.15751 | —         |
| Roman     | F087 ¹       |  26.302 | GAUSSIAN |     0.073 |       0.110 |      85.0 |        0.11480 | —         |
| Roman     | F106 ¹       |  26.355 | GAUSSIAN |     0.087 |       0.110 |     107.0 |        0.03834 | —         |
| Roman     | F129 ¹       |  26.353 | GAUSSIAN |     0.106 |       0.110 |     107.0 |        0.03833 | —         |
| Roman     | F158 ¹       |  26.376 | GAUSSIAN |     0.128 |       0.110 |     107.0 |        0.03791 | —         |
| Roman     | F184 ¹       |  25.912 | GAUSSIAN |     0.146 |       0.110 |     409.0 |        0.01401 | —         |
| Roman     | F146 ¹       |  27.584 | GAUSSIAN |     0.105 |       0.110 |      46.8 |        0.00136 | —         |
| ZTF       | g            |  26.325 | GAUSSIAN |     2.100 |       1.010 |      30.0 |        0.21950 | —         |
| ZTF       | r            |  26.275 | GAUSSIAN |     2.000 |       1.010 |      30.0 |        0.31357 | —         |
| ZTF       | i            |  25.660 | GAUSSIAN |     2.100 |       1.010 |      30.0 |        0.41918 | —         |

---

¹ **Roman survey modes** — each filter requires a matched `survey_mode` in lenstronomy:
F062, F087 → `time_domain_wide`; F106, F129, F158 → `wide_area`; F184 → `time_domain_deep`; F146 → `microlensing`.
The `t_exp` values above reflect the chosen mode.

² **pixel_PSF = ✓** — `psf_type='PIXEL'` is configured, but the PSF kernel map must be provided
externally at runtime. It is not bundled in lenstronomy.

---

## Catalog magnitude → Sersic amplitude

### What the catalog gives you

The Qiuhan catalog stores **apparent AB magnitudes** (e.g. `source_app_mag_VIS`, `deflector_app_mag_VIS`). gwemfish/herculens light profiles are parameterised by an **amplitude** `amp`, which sets the peak surface brightness in units of e⁻/s/pixel². The conversion has two steps.

### Step 1 — magnitude to counts per second

The AB zero-point definition gives the total observed flux of the source:

$$F [\text{e}^- / \text{s}] = 10^{(ZP - m) / 2.5}$$

where $m$ is the apparent magnitude and $ZP$ is the survey zero-point (e.g. 25.72 for Euclid VIS).  
This is what `lenstronomy.Util.data_util.magnitude2cps` computes.

### Step 2 — flux to profile amplitude

A Sersic profile with `amp = 1` integrates to a total flux $F_\text{norm}$ over the image plane (computed numerically by lenstronomy with `total_flux(kwargs, norm=True)`). The required amplitude is then:

$$\boxed{\texttt{amp} = \frac{F}{F_\text{norm}} = \frac{10^{(ZP - m) / 2.5}}{F_\text{norm}}}$$

$F_\text{norm}$ depends on the profile shape — `R_sersic`, `n_sersic`, `e1`, `e2` — so the same magnitude gives a different `amp` for a compact vs. extended galaxy.

### Code

`scripts/common.py` handles this via `MagAmpConversion`, which wraps both steps:

```python
from lenstronomy.SimulationAPI.mag_amp_conversion import MagAmpConversion
from scripts.common import EUCLID_VIS_ZP  # 25.72

kwargs_model = {
    "lens_light_model_list": ["SERSIC_ELLIPSE"],
    "source_light_model_list": ["SERSIC_ELLIPSE"],
}
mag_converter = MagAmpConversion(kwargs_model=kwargs_model, magnitude_zero_point=EUCLID_VIS_ZP)

kwargs_lens_light_mag = [{
    "magnitude": row["deflector_app_mag_VIS"],   # apparent AB mag from catalog
    "R_sersic":  row["deflector_Re"],
    "n_sersic":  4.0,
    "e1": lens_e1, "e2": lens_e2,
    "center_x": 0.0, "center_y": 0.0,
}]
kwargs_source_mag = [{
    "magnitude": row["source_app_mag_VIS"],
    "R_sersic":  row["source_Re"],
    "n_sersic":  row["source_sersic_index"],
    "e1": source_e1, "e2": source_e2,
    "center_x": row["source_relative_x"],
    "center_y": row["source_relative_y"],
}]

# Returns (kwargs_lens_light, kwargs_source, kwargs_ps) with 'magnitude' replaced by 'amp'
lens_light_kwargs, source_kwargs, _ = mag_converter.magnitude2amplitude(
    kwargs_lens_light_mag=kwargs_lens_light_mag,
    kwargs_source_mag=kwargs_source_mag,
)

amp_lens  = lens_light_kwargs[0]["amp"]   # e⁻/s/arcsec², ready for gwemfish
amp_source = source_kwargs[0]["amp"]
```

### Why not just use `magnitude2cps` directly?

`magnitude2cps(m, ZP)` gives the **total flux** of the source. But gwemfish needs `amp`, the **peak** of the normalised profile. Dividing by $F_\text{norm}$ (the profile integral at `amp=1`) converts total flux into the correct peak amplitude. Skipping that step would make all sources too faint by a factor of $F_\text{norm}$, which varies with size and Sérsic index.
