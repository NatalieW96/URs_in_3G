import numpy as np
import bilby
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns

def compute_lambda_confidence_intervals_spectral(
    ms, res,
    lower_bounds=[0, 5, 25],
    upper_bounds=[100, 95, 75],
    min_samples=20
):
    """
    Computes confidence intervals (including 100%) for radius at fixed mass values.
    Interpolates radius(mass) for each EOS sample and returns smoothed, masked intervals.

    Parameters:
        ms (np.ndarray): Array of mass values (M_sun) to evaluate.
        res: Object with .posterior attribute (DataFrame of EOS parameters).
        generator: Object with .generate_macro_eos(params) method.
        lower_bounds: Lower percentiles for CI.
        upper_bounds: Upper percentiles for CI.
        min_samples: Minimum number of valid samples to include a mass point.
        smoothing_sigma: Sigma for Gaussian smoothing of CI curves.

    Returns:
        dict with keys:
            'masses' (ms),
            'radius_ci' (dict of CIs),
            'valid_mask' (boolean array of shape ms)
    """
    n_samples = len(res.posterior)
    n_masses = len(ms)

    all_lambdas_at_m = np.full((n_samples, n_masses), np.nan)

    for i in range(n_samples):
        params = np.array([res.posterior.iloc[i][0], res.posterior.iloc[i][1], res.posterior.iloc[i][2], res.posterior.iloc[i][3]])
        try:
            g0, g1, g2, g3 = bilby.gw.conversion.spectral_pca_to_spectral(*params)
            eos = bilby.gw.utils.lalsim_SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
            family = bilby.gw.utils.lalsim_CreateSimNeutronStarFamily(eos)
            min_mass = bilby.gw.utils.lalsim_SimNeutronStarFamMinimumMass(family) / bilby.core.utils.constants.solar_mass
            max_mass = bilby.gw.utils.lalsim_SimNeutronStarMaximumMass(family) /  bilby.core.utils.constants.solar_mass
            masses = np.linspace(max(min_mass+0.05, 1), max_mass-0.05, 1000)
            radii = np.array([bilby.gw.utils.lalsim_SimNeutronStarRadius(mass_i * bilby.core.utils.constants.solar_mass, family)for mass_i in masses])
            love_numbers = np.array([bilby.gw.utils.lalsim_SimNeutronStarLoveNumberK2(mass_i * bilby.core.utils.constants.solar_mass, family)for mass_i in masses])
            mass_geometrized = masses * bilby.core.utils.constants.solar_mass *  bilby.core.utils.constants.gravitational_constant / bilby.core.utils.constants.speed_of_light ** 2.
            compactness = mass_geometrized /radii
            lambdas = (2./3.) * love_numbers * (1./(compactness**5))
            lambda_interp = interp1d(masses, lambdas, bounds_error=False, fill_value=np.nan)
            all_lambdas_at_m[i, :] = lambda_interp(ms)
            print(f"Processed sample {i+1}/{n_samples}", end='\r')
        except Exception:
            continue
    valid_counts = np.sum(~np.isnan(all_lambdas_at_m), axis=0)
    valid_mask = valid_counts >= min_samples
    #plt.plot(soft_radius_interp(ms), ms, c='k', label='Soft EOS')
    lambda_ci = {}
    for low, high in zip(lower_bounds, upper_bounds):
        key = f"{high - low}"
        lower = np.nanpercentile(all_lambdas_at_m, low, axis=0)
        upper = np.nanpercentile(all_lambdas_at_m, high, axis=0)
        lambda_ci[key] = (lower, upper)
    lambda_ci["median"] = np.nanpercentile(all_lambdas_at_m, 50, axis=0)
    return {
        'masses': ms,
        'lambda_ci': lambda_ci,
        'valid_mask': valid_mask
    }

# Define mass grid
ms = np.linspace(1.0, 2.2, 1000)

# Color and opacity settings
c = sns.husl_palette(3)
colors = {'100': c[0], '90': c[0], '50': c[0]}
opacity = {'100': 0.2, '90': 0.4, '50': 0.6}

rundir = '/work/williams5/testing-urs/bilby/runs'


res_soft = bilby.result.read_in_result(filename=f'{rundir}/samplefUR/EOS_UR/soft/nuclear_sampling_spectral/nuclear_sampling_spectral_result.json')

intervals_soft = compute_lambda_confidence_intervals_spectral(ms, res_soft)

np.save(f'{rundir}/samplefUR/EOS_UR/soft/nuclear_sampling_spectral/nuclear_sampling_spectral_intervals_lambda.npy', intervals_soft)

#res_stiff = bilby.result.read_in_result(filename=f'{rundir}/samplefUR/EOS_UR/stiff/nuclear_sampling_spectral/nuclear_sampling_spectral_result.json')

#intervals_stiff = compute_lambda_confidence_intervals_spectral(ms, res_stiff)

#np.save(f'{rundir}/samplefUR/EOS_UR/stiff/nuclear_sampling_spectral/nuclear_sampling_spectral_intervals_lambda.npy', intervals_stiff)

