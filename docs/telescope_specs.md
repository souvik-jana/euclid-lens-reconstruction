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

The Qiuhan catalog stores **apparent AB magnitudes** (e.g. `source_app_mag_VIS`, `deflector_app_mag_VIS`). gwemfish/herculens light profiles are parameterised by an **amplitude** `amp`, which is the surface brightness **at the half-light radius** $R_\text{sersic}$ in units of e⁻/s/pixel². The conversion has two steps.

### Step 1 — magnitude to counts per second

The AB zero-point definition gives the total observed flux of the source:

$$F [\text{e}^- / \text{s}] = 10^{(ZP - m) / 2.5}$$

where $m$ is the apparent magnitude and $ZP$ is the survey zero-point (e.g. 25.72 for Euclid VIS).  
This is what `lenstronomy.Util.data_util.magnitude2cps` computes.

### Step 2 — flux to profile amplitude

The lenstronomy `SERSIC_ELLIPSE` profile (verified from source) is:

$$I(R) = I_e \exp\left( -b_n \left[ \left(\frac{R}{R_e}\right)^{1/n} - 1 \right] \right)$$

where `amp` $= I_e$ and $b_n$ is the approximation to the exact half-light condition $2\gamma(2n, b_n) = \Gamma(2n)$, implemented in lenstronomy as:

$$b_n = 1.9992 \cdot n - 0.3271$$

At $R = R_e$ (= `R_sersic` in lenstronomy) the exponent vanishes and $I = I_e =$ `amp`, confirming that `amp` is the surface brightness **at the half-light radius** — not the central peak. The central peak (at $R=0$) is:

$$I(0) = I_e \cdot e^{b_n}$$

For $n=4$ (de Vaucouleurs): $b_4 = 1.9992 \times 4 - 0.3271 \approx 7.67$, so $I(0) \approx 2150 \times I_e$.

A Sersic profile with `amp = 1` integrates to a total flux $F_\text{norm}$ over the image plane (computed numerically by lenstronomy with `total_flux(kwargs, norm=True)`). The required amplitude is then:

$$I_e = \frac{F}{F_\text{norm}} = \frac{10^{(ZP - m) / 2.5}}{F_\text{norm}}$$

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

amp_lens   = lens_light_kwargs[0]["amp"]   # surface brightness at R_sersic [e⁻/s/pixel²], ready for gwemfish
amp_source = source_kwargs[0]["amp"]
```

### Why not just use `magnitude2cps` directly?

`magnitude2cps(m, ZP)` gives the **total flux** of the source. But gwemfish needs `amp`, the **peak** of the normalised profile. Dividing by $F_\text{norm}$ (the profile integral at `amp=1`) converts total flux into the correct peak amplitude. Skipping that step would make all sources too faint by a factor of $F_\text{norm}$, which varies with size and Sérsic index.

---

## Background noise (`bkg_rms`)

### Camera properties

All values sourced live from `kwargs_single_band()`. `num_exposures` varies per band (survey mode / coadd strategy), so this table is per-band.

| Telescope | Band         | read_noise (e⁻) | pixel_scale (") | ccd_gain (e⁻/ADU) | sky_brightness (mag/arcsec²) | num_exposures |
|-----------|--------------|----------------:|----------------:|-------------------:|-----------------------------:|--------------:|
| Euclid    | VIS          |            4.20 |           0.101 |               3.10 |                         22.3 |             4 |
| Euclid    | Y            |            6.10 |           0.300 |               3.10 |                         22.1 |             4 |
| Euclid    | J            |            6.10 |           0.300 |               3.10 |                         22.2 |             4 |
| Euclid    | H            |            6.10 |           0.300 |               3.10 |                         22.3 |             4 |
| HST       | WFC3_F160W   |            4.00 |           0.080 |               2.50 |                         22.3 |             1 |
| HST       | TDLMC_F160W  |            4.00 |           0.080 |               2.50 |                         22.0 |             1 |
| JWST      | F115W        |           15.77 |           0.031 |               2.05 |                         30.96 |            8 |
| JWST      | F150W        |           15.77 |           0.031 |               2.05 |                         29.96 |            8 |
| JWST      | F200W        |           15.77 |           0.031 |               2.05 |                         29.52 |            1 |
| JWST      | F277W        |           13.25 |           0.063 |               1.82 |                         28.96 |            8 |
| JWST      | F356W        |           13.25 |           0.063 |               1.82 |                         28.39 |            1 |
| JWST      | F444W        |           13.25 |           0.063 |               1.82 |                         28.15 |            8 |
| LSST      | u            |           10.00 |           0.200 |               2.30 |                         22.99 |          140 |
| LSST      | g            |           10.00 |           0.200 |               2.30 |                         22.26 |          200 |
| LSST      | r            |           10.00 |           0.200 |               2.30 |                         21.20 |          460 |
| LSST      | i            |           10.00 |           0.200 |               2.30 |                         20.48 |          460 |
| LSST      | z            |           10.00 |           0.200 |               2.30 |                         19.60 |          400 |
| LSST      | y            |           10.00 |           0.200 |               2.30 |                         18.61 |          400 |
| DES       | g            |            7.00 |           0.263 |               4.00 |                         22.01 |           10 |
| DES       | r            |            7.00 |           0.263 |               4.00 |                         21.15 |           10 |
| DES       | i            |            7.00 |           0.263 |               4.00 |                         19.89 |           10 |
| DES       | z            |            7.00 |           0.263 |               4.00 |                         18.72 |           10 |
| DES       | Y            |            7.00 |           0.263 |               4.00 |                         17.96 |           10 |
| Roman     | F062         |            8.50 |           0.110 |               1.00 |                         23.19 |             1 |
| Roman     | F087         |            8.50 |           0.110 |               1.00 |                         22.93 |             1 |
| Roman     | F106         |            8.50 |           0.110 |               1.00 |                         22.99 |             6 |
| Roman     | F129         |            8.50 |           0.110 |               1.00 |                         22.99 |             6 |
| Roman     | F158         |            8.50 |           0.110 |               1.00 |                         23.10 |             6 |
| Roman     | F184         |            8.50 |           0.110 |               1.00 |                         23.22 |             4 |
| Roman     | F146         |            8.50 |           0.110 |               1.00 |                         22.03 |         41000 |
| ZTF       | g            |           10.30 |           1.010 |               5.80 |                         22.01 |            40 |
| ZTF       | r            |           10.30 |           1.010 |               5.80 |                         21.15 |            40 |
| ZTF       | i            |           10.30 |           1.010 |               5.80 |                         19.89 |            40 |

### What it is

`SingleBand.background_noise` combines read noise and sky shot noise into a single per-pixel noise floor, normalised to per-second:

### Worked example — Euclid VIS

Using values from the camera-properties table above and the `bkg_rms` column (expected result: **0.01090 e⁻/s/pixel**):

Throughout this section, `num_exposures` (column name) is written as $n_\text{exp}$ in formulas and `num_exposures` in code — all the same quantity.

| Step | Quantity | Calculation | Result |
|------|----------|-------------|--------|
| 1 | Total exposure time | $t_\text{tot} = \text{num\_exposures} \times t_\text{exp} = 4 \times 566$ | 2264 s |
| 2 | Sky flux density | $F_\text{sky} = 10^{(25.72 - 22.3)/2.5}$ | 23.34 e⁻/s/arcsec² |
| 3 | Sky flux per pixel | $F_\text{sky} \times \Delta^2 = 23.34 \times 0.101^2$ | 0.2381 e⁻/s/pixel |
| 4 | Total sky electrons | $t_\text{tot} \times 0.2381 = 2264 \times 0.2381$ | 538.8 e⁻/pixel |
| 5 | Read noise variance | $\text{num\_exposures} \times \text{read\_noise}^2 = 4 \times 4.2^2$ | 70.56 e²/pixel |
| 6 | Total variance | $70.56 + 538.8$ | 609.4 e²/pixel |
| 7 | **$\sigma_\text{bkg}$** | $\sqrt{609.4} / 2264$ | **0.01090 e⁻/s/pixel** ✓ |

In code this is `data_util.bkg_noise(read_noise, t_exp, sky_brightness_cps, pixel_scale, num_exposures)`:

```python
import numpy as np

ZP, m_sky = 25.72, 22.3
t_exp, num_exposures, read_noise, pix = 566.0, 4, 4.2, 0.101

t_tot         = num_exposures * t_exp                  # 2264 s
F_sky         = 10**((ZP - m_sky) / 2.5)              # e⁻/s/arcsec²    (23.34)
sky_per_pixel = F_sky * pix**2                         # e⁻/s/pixel      (0.2381)
sky_tot       = t_tot * sky_per_pixel                  # e⁻/pixel        (538.8)
rn_tot        = num_exposures * read_noise**2          # e²/pixel        (70.56)

sigma_bkg = np.sqrt(rn_tot + sky_tot) / t_tot         # e⁻/s/pixel  ->  0.01090
```

The general formula (with `num_exposures` included):

$$\sigma_\text{bkg} = \frac{\sqrt{\text{num\_exposures} \times \text{read\_noise}^2 + \text{num\_exposures} \times t_\text{exp} \times F_\text{sky} \times \Delta^2}}{\text{num\_exposures} \times t_\text{exp}}$$

where:

$$F_\text{sky} = 10^{(ZP - m_\text{sky}) / 2.5}$$ &nbsp;&nbsp; [e⁻/s/arcsec²]

$m_\text{sky}$ is the `sky_brightness` column (mag/arcsec²), $\Delta$ is `pixel_scale`, and `read_noise` is in e⁻. For Euclid VIS these give $\sigma_\text{bkg} \approx 0.011$ e⁻/s/pixel.

### Using `bkg_rms` in gwemfish

gwemfish's `noise_simu_kwargs["background_rms"]` expects e⁻/s/pixel — the same unit `SingleBand.background_noise` returns — so it drops in directly with no conversion.

```python
from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
from lenstronomy.SimulationAPI.observation_api import SingleBand

cfg_tel = Euclid("VIS", "GAUSSIAN").kwargs_single_band()
bkg_rms = SingleBand(**cfg_tel).background_noise   # 0.011 e⁻/s/pixel
t_exp   = cfg_tel["exposure_time"]                 # 566.0 s

noise_simu_kwargs = {
    "npix": 40,
    "background_rms": bkg_rms,   # use directly — no conversion needed
    "exposure_time": t_exp,
}
```

### Using `bkg_rms` in PyAutoLens (PAL)

PAL's `SimulatorImaging` takes `background_sky_level` in **e⁻/pixel** (total electrons, not per second). You must convert:

$$\sigma_\text{bkg}^2 \times t_\text{exp}$$ &nbsp;&nbsp; [e⁻/pixel]

The two variances are identical — this is only a unit reframing.

```python
import autolens as al

background_sky_level = bkg_rms**2 * t_exp   # e⁻/pixel  (PAL convention)

sim = al.SimulatorImaging(
    exposure_time=t_exp,
    psf=psf,
    background_sky_level=background_sky_level,
    add_poisson_noise_to_data=True,
    noise_seed=seed,
)
```

### Quick reference

| Framework | Input key | Formula | Unit |
|-----------|-----------|---------|------|
| gwemfish  | `background_rms` | $\sigma_\text{bkg}$ | e⁻/s/pixel |
| PAL       | `background_sky_level` | $\sigma_\text{bkg}^2 \times t_\text{exp}$ | e⁻/pixel |
