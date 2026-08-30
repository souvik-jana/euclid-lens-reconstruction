# ==========================================================
#set up and imports
# ==========================================================
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=12")
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
print("JAX devices:", jax.devices())

from pathlib import Path

import matplotlib.pyplot as plt

from gwemfish.simple_pipeline import (
    _deep_merge_dict,
    make_default_cfg,
    plot_system_observation,
    setup_em_observation,
    setup_gw_observation,
    run_inference, 
    plot_posterior, 
    to_source_plane_samples, 
    plot_source_posterior,
)
import numpy as np
import copy
import json

from scripts.common import sample_cfg, row_to_cfg, NumpyEncoder

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================================
#Import Qiuhan's catalog
# ==========================================================
import pandas as pd
repo_root = Path(__file__).resolve().parent.parent
df = pd.read_csv(repo_root / "catalog" / "filtered_lens_catalog_PL_IC_gt_70.csv")


# ==========================================================
#Searching for 4 image candidates (roughly)
# ==========================================================
import time
start_time = time.time()

accepted_rows = []
for i in range(200, 1000):

    one_galaxy = df.iloc[i:i+1]

    if one_galaxy.iloc[0]['deflector_thetaE'] < 0.6:
        continue

    row = one_galaxy.iloc[0]
    cfg = row_to_cfg(row, sample_cfg, False)
    cfg["use_parameter_layout"] = True
    ctx = setup_em_observation(cfg=cfg)
    tp_pair = ctx['truth_params']

    if len(tp_pair['x_image_true_em']) < 4:
        continue
    
    #print(f'System {i} is close enough to the GW source and has at least four images. Proceed with manual inspection')
    accepted_rows.append(one_galaxy.iloc[0])

candidates = pd.DataFrame(accepted_rows, columns=df.columns)
print("--- %s seconds ---" % (time.time() - start_time))


# ==========================================================
#preparing to compute the posterior overlap and EM, GW only evidnces
# ==========================================================

from scipy.stats import gaussian_kde
shared_params = ['lens0_e1', 'lens0_e2', 'lens0_gamma', 'lens0_theta_E', 'lens1_gamma1', 'lens1_gamma2']
import numpyro.distributions as dist
from gwemfish import prune_gw_images
from scipy.stats import norm
from scipy.special import logsumexp
start = 2
finish = 4
this_run = candidates.iloc[start:finish]


# ==========================================================
#run GWEMFISH inference and compute posterior overlap. Then, save as jsonl
# ==========================================================
for idx, row in this_run.iterrows():

    print(f'----------------------------------------Starting {idx}-th iteration---------------------------------------------------')
    
    #row = candidates.iloc[i]
    cfg_org = row_to_cfg(row, sample_cfg, True)
    cfg_org['em']['pixel_grid_kwargs']['npix'] = 60
    cfg_org['em']['noise_simu_kwargs']['npix'] = 60
    cfg_org['em']['noise_inf_kwargs']['npix'] = 60
    cfg_org['em']['psf_kwargs']['fwhm'] = 0.18
    
    cfg_org["use_parameter_layout"] = True

    ctx_post_em = setup_em_observation(cfg=cfg_org)
    ctx = setup_gw_observation(ctx_post_em, cfg=cfg_org)
    
    
    if len(ctx["x_img_gw"]) < 4:
        print('-----------END: LESS THAN 4 IMAGES----------')
        continue
        
    if len(ctx["x_img_gw"]) > 4:
        ctx = prune_gw_images(ctx, n_keep=4)
    
    #save true values
    tp = ctx['truth_params']
    fig = plot_system_observation(
        ctx,
        cfg={"output": {"save_system_plot_path": os.path.join(OUTPUT_DIR, f"system_observation_{idx}.png")}},
    )

    ctx["cfg"]["priors"] = {
        "T_star": float(tp["T_star"]),
        "dL": float(tp["dL"]),
        'lens0_gamma': dist.TruncatedNormal(tp['lens0_gamma'], 0.1, low=0.001, high=5.0),
        'lens0_theta_E': dist.TruncatedNormal(tp['lens0_theta_E'], 0.1, low=0.001, high=5.0),
         #'lens0_e1': float(tp["lens0_e1"]), 
         #'lens0_e2': float(tp['lens0_e2']),
         'lens0_e1': dist.Normal(tp['lens0_e1'], 0.01),
         'lens0_e2': dist.Normal(tp['lens0_e2'], 0.01),
         'lens0_center_x': 0.0,
         'lens0_center_y': 0.0,
         #'lens1_gamma1': float(tp["lens1_gamma1"]),
         #'lens1_gamma2': float(tp["lens1_gamma2"]),
         'lens1_gamma1': dist.Normal(tp['lens1_gamma1'], 0.01),
         'lens1_gamma2': dist.Normal(tp['lens1_gamma2'], 0.01),
         'lens1_ra_0': 0.0,
         'lens1_dec_0': 0.0,
          'light0_R_sersic': float(tp["light0_R_sersic"]),
          'light0_n_sersic': float(tp["light0_n_sersic"]),
          'light0_amp': float(tp['light0_amp']),
         'light0_e1': float(tp["light0_e1"]),
         'light0_e2': float(tp["light0_e2"]),
         'light0_center_x': float(tp["light0_center_x"]),
         'light0_center_y': float(tp["light0_center_y"]),
          "noise_sigma_bkg": tp["noise_sigma_bkg"],  
        } 

    ctx_em = copy.deepcopy(ctx)
    ctx_gw = copy.deepcopy(ctx)
    
    
    #perform the parameter estimation in EM-only mode
    method = 'deriv-approx'
    
    samples_em, truths_em = run_inference(
    ctx_em,
    mode="EM-only",#"EM-only",#"GW-only",
    method=method,
    cfg={
        "output": {"json_tag": method},
        **({"inference": {"informed": True, "regularize": False,"num_chains": 8, "num_samples": 14500, "num_warmup": 9500}} if method == "deriv-approx" else {}),
        # "inference": {"num_chains": 12, "num_samples": 8000, "num_warmup": 8000, "n_fisher_samples": 10000},
    },
    )

    #Param est for GW-only mode
    
    samples_gw, truths_gw = run_inference(
    ctx_gw,
    mode="GW-only",#"EM-only",#"GW-only",
    method=method,
    cfg={
        "output": {"json_tag": method},
        **({"inference": {"informed": True, "regularize": False,"num_chains": 8, "num_samples": 14500, "num_warmup": 9500}} if method == "deriv-approx" else {}),
        # "inference": {"num_chains": 12, "num_samples": 8000, "num_warmup": 8000, "n_fisher_samples": 10000},
    },
    )

    #param sigmas and check if NaN values exists
    has_any_nan_em = any(np.isnan(arr).any() for arr in samples_em.values())
                
    has_any_nan_gw = any(np.isnan(arr).any() for arr in samples_gw.values())
    
    if has_any_nan_gw or has_any_nan_em:
        print("--------------------UH OH! Parameter estimation exploded!!!! Contains NaN values-----------------")
        break

    print(f'-------------------Finished parameter estimation -------------------------')
    
    #compute the em only evidence
    logp0_em = float(ctx_em["fisher"]["logp0"]) #true point, 
    H_em = np.asarray(ctx_em["fisher"]["H0"]) # also around true point
    
    k_em = H_em.shape[0]
    
    sign_em, logdet_em = np.linalg.slogdet(-H_em) #Compute the sign and (natural) logarithm of the determinant of an array.
    
    logZ_em = (
        logp0_em
        + 0.5 * k_em * np.log(2*np.pi)
        - 0.5 * logdet_em
    )


    samples_matrix_GW = np.vstack([samples_gw[p] for p in shared_params])
    #defining the gaussian kde
    kde_gw = gaussian_kde(samples_matrix_GW)
    
    print(f'-------------------Finished computing logZ_em-------------------------')
    
    #compute the joint evidence with monte carlo integration
    samples_matrix_EM = np.vstack([samples_em[p] for p in shared_params])
    
    p_gw = kde_gw(samples_matrix_EM)
    
    gamma_samples = np.asarray(samples_em["lens0_gamma"])
    thetaE_samples = np.asarray(samples_em["lens0_theta_E"])
    e1_samples = np.asarray(samples_em["lens0_e1"])
    e2_samples = np.asarray(samples_em["lens0_e2"])
    gamma1_samples = np.asarray(samples_em["lens1_gamma1"])
    gamma2_samples = np.asarray(samples_em["lens1_gamma2"])
    
    log_prior = (
        norm.logpdf(gamma_samples,tp['lens0_gamma'],0.1)
        + norm.logpdf(thetaE_samples,tp['lens0_theta_E'],0.1)
        + norm.logpdf(e1_samples,tp['lens0_e1'],0.01)
        + norm.logpdf(e2_samples,tp['lens0_e2'],0.01)
        + norm.logpdf(gamma1_samples,tp['lens1_gamma1'],0.01)
        + norm.logpdf(gamma2_samples,tp['lens1_gamma2'],0.01)
    )
    
    log_integrand = np.log(p_gw) - log_prior
    
    log_I = logsumexp(log_integrand) - np.log(len(log_integrand))
    
    I = np.exp(log_I)

    print(f'-------------------Finished evaluating the integral -------------------------')

    result_dict =  {'index': idx,
                    'truths_em': truths_em, 
                    'logZ_em': logZ_em,
                    'I': I,
                    'logZ_em+log_I': (logZ_em + log_I)}
    
    #sample_truth_pairs.append(result)
    
    # 2. Append directly to the file immediately
    with open('progress_results.jsonl', 'a') as f:
        json_line = json.dumps(result_dict, cls=NumpyEncoder)
        f.write(json_line + '\n')

    print(f'----------------------------------------Finished {idx}-th iteration---------------------------------------------------')

# ==========================================================
#Load the results and plot the EM improvements
# ==========================================================

import json

# Initialize an empty list to store the recovered data
loaded_run = []

# Open and read the file line by line
with open('progress_results.jsonl', 'r') as f:
    for line in f:
        if line.strip():  # This skips any accidental empty lines
            loaded_run.append(json.loads(line))

print(f"Successfully loaded {len(loaded_run)} items from the file!")

EM_contributions = []
for i in range(len(loaded_run)):
    print(loaded_run[i]['I'])
    EM_contributions.append(loaded_run[i]['I'])


plt.hist(EM_contributions)
plt.show()
