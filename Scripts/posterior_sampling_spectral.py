import json
import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity
import bilby
import matplotlib.pyplot as plt

import argparse
from bilby.core.likelihood import Likelihood
import sys

from bilby.gw.conversion import (
    chirp_mass_and_mass_ratio_to_component_masses,
    lambda_1_lambda_2_to_lambda_tilde,
    lambda_1_lambda_2_to_delta_lambda_tilde
    )

class QURLikelihood(Likelihood):
    def __init__(self, posterior_files, mass_points = 32, parameters = {}):
        kde_list = []
        comp_mass_list = []
        differentials_list = []
        for i, post in enumerate(posterior_files):
            with open(post, 'r') as f:
                post_data = json.load(f)
                post_data = post_data['posterior']['content']
            chirp_mass = post_data['chirp_mass_source']
            mass_ratio = post_data['mass_ratio']

            tidal_defo = post_data['lambda_tilde']
            delta_lambda = post_data['delta_lambda_tilde']

            scale = 10000
            ratio = len(tidal_defo) // scale if len(tidal_defo) > scale  else 1

            kde_data = (chirp_mass[::ratio], mass_ratio[::ratio],
                        tidal_defo[::ratio], delta_lambda[::ratio])
            kde = KernelDensity(kernel='gaussian').fit(np.vstack(kde_data).T)
            # kde = gaussian_kde(kde_data)
            kde_list.append(kde)

            mc_low, mc_high = np.percentile(chirp_mass, [2.5, 97.5])
            q_low, q_high = np.percentile(mass_ratio, [2.5, 97.5])
            test_mc, dmc = np.linspace(mc_low, mc_high, mass_points, retstep=True)
            test_q, dq = np.linspace(q_low, q_high, mass_points, retstep=True)
            differentials_list.append(dmc * dq)

            mcs, qs = np.meshgrid(test_mc, test_q)


            m1, m2 = chirp_mass_and_mass_ratio_to_component_masses(mcs.flatten(), qs.flatten())
            idx = (m2 >=1.)
            m1, m2, mc, q = m1[idx], m2[idx], mcs.flatten()[idx], qs.flatten()[idx]


            comp_mass_list.append((m1, m2, mc, q))

        self.kde_list = kde_list
        self.comp_mass_list = comp_mass_list
        self.differentials_list = differentials_list
        self.test_points = mass_points
        self.base_mass_grid = np.linspace(0,1,100)
        print('initiation finished')

        super().__init__(parameters)

    def log_likelihood(self):

        # EOS checks
        eos_spectral_pca_gamma_0 = float(self.parameters["eos_spectral_pca_gamma_0"])
        eos_spectral_pca_gamma_1 = float(self.parameters["eos_spectral_pca_gamma_1"])
        eos_spectral_pca_gamma_2 = float(self.parameters["eos_spectral_pca_gamma_2"])
        eos_spectral_pca_gamma_3 = float(self.parameters["eos_spectral_pca_gamma_3"])

        g0, g1, g2, g3 = bilby.gw.conversion.spectral_pca_to_spectral(eos_spectral_pca_gamma_0, 
                                                  eos_spectral_pca_gamma_1,
                                                  eos_spectral_pca_gamma_2,
                                                  eos_spectral_pca_gamma_3)



        try:
            if bilby.gw.utils.lalsim_SimNeutronStarEOS4ParamSDGammaCheck(g0,g1,g2,g3) != 0:
                return -np.inf
            eos = bilby.gw.utils.lalsim_SimNeutronStarEOS4ParameterSpectralDecomposition(
                g0, g1, g2, g3)
        
            family = bilby.gw.utils.lalsim_CreateSimNeutronStarFamily(eos)
            max_pseudo_enthalpy = bilby.gw.utils.lalsim_SimNeutronStarEOSMaxPseudoEnthalpy(eos)
            max_speed_of_sound = bilby.gw.utils.lalsim_SimNeutronStarEOSSpeedOfSoundGeometerized(max_pseudo_enthalpy, eos)
            min_mass = bilby.gw.utils.lalsim_SimNeutronStarFamMinimumMass(family) / bilby.core.utils.constants.solar_mass
            max_mass = bilby.gw.utils.lalsim_SimNeutronStarMaximumMass(family) /  bilby.core.utils.constants.solar_mass

            if max_speed_of_sound > 1.1 or min_mass > 1.2 or max_mass <  1.8 or max_mass > 3.0: #CHECK THESE VALUES
                return -np.inf

            # Generate lambdas
            mass_range = min_mass + 0.05 + self.base_mass_grid * (max_mass - min_mass - 0.1)
            lam_data = np.array([bilby.gw.conversion.lambda_from_mass_and_family(mass, family) for mass in mass_range])

        except RuntimeError:
            return -np.inf

        log_likelihood = 0.0
        for lambda_kde, (m1, m2, mc, q), dA in zip(self.kde_list, self.comp_mass_list, self.differentials_list):

            # Calculate the log likelihood for each posterior, assuming bh beyond mtov
            lam1, lam2 = np.interp([m1 , m2], mass_range, lam_data, right=0, left=0)
            lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(lam1, lam2, m1, m2)
            delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(lam1, lam2, m1, m2)

            # kde_values = lambda_kde.logpdf((lambda_tilde, delta_lambda_tilde))
            kde_values = lambda_kde.score_samples(np.vstack((mc, q, lambda_tilde, delta_lambda_tilde)).T)
            log_likelihood += logsumexp(kde_values) + dA  # logsumexp to account for the differentials
        # breakpoint()
        return log_likelihood


def resample(posteriors, prior_file, nlive=1000, outdir=None, label=None, **kwargs):

    priors = bilby.core.prior.PriorDict(filename=prior_file)
    likelihood = QURLikelihood(posteriors)

    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        # sampler=sampler,
        # conversion_function = likelihood.conversion_function,
        nlive=nlive,
        outdir=outdir,
        label=label,
        npool=196,
        **kwargs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resample posterior files with a new EOS generator.")
    parser.add_argument("--posterior-files", nargs="*", help="List of posterior files to resample.")
    parser.add_argument("--prior-file", type=str, required=True, help="Path to the prior file.")
    #parser.add_argument("--eos-gen-metadata", type=str, required=True, help="Path to the EOS generator metadata file.")
    parser.add_argument("--nlive", type=int, default=300, help="Number of live points for the sampler.")
    parser.add_argument("--outdir", type=str, default="outdir", help="Output directory for the results.")
    parser.add_argument("--label", type=str, default="resampled", help="Label for the output files.")

    args = parser.parse_args()

    
    resample(args.posterior_files, args.prior_file, nlive=args.nlive, outdir=args.outdir, label=args.label)
