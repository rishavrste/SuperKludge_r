
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from itertooåls import product
import os
import h5py
from tqdm import tqdm
import pandas as pd

#few utils
from few.utils.utility import get_p_at_t
from few.utils.constants import MTSUN_SI
from few.utils.geodesic import get_fundamental_frequencies
#few trajectory
from few.trajectory.inspiral import EMRIInspiral
from few.trajectory.ode.flux import SuperKludgeFlux
#few waveform
from few.waveform import FastKerrEccentricEquatorialFlux, GenerateEMRIWaveform
from few.waveform.waveform import SuperKludgeWaveform
from few.utils.constants import YRSID_SI

#sef imports
from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import generate_PSD, padding, inner_product
from stableemrifisher.fisher.derivatives import derivative
from stableemrifisher.fisher.stablederivative import StableEMRIDerivative
from stableemrifisher.noise import sensitivity_LWA

#lisa-on-gpu import
from fastlisaresponse import ResponseWrapper  # Response function 

#LISAanalysistools imports
from lisatools.detector import ESAOrbits, EqualArmlengthOrbits #ESAOrbits correspond to esa-trailing-orbits.h5, EqualArmlengthOrbits are equalarmlength-orbits.h5
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens, T1TDISens
from lisatools.sensitivity import get_sensitivity,CornishLISASens

use_gpu = True
from parismc.sampler import SamplerConfig
from parismc.sampler import Sampler

from smt.sampling_methods import LHS

if not use_gpu:
    
    import few
    
    #tune few configuration
    cfg_set = few.get_config_setter(reset=True)
    
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info");
else:
    pass #let the backend decide for itself

#waveform class setup
waveform_class = SuperKludgeWaveform
max_step_days = 10.0 #max trajectory step size in days
inspiral_kwargs = {
    "err":1e-11, #default = 1e-11
    "max_step_size":max_step_days*24*60*60, #in seconds
    "use_gpu":use_gpu
}
sum_kwargs = {
    "pad_output": True, # True if expecting waveforms smaller than LISA observation window.
  #  "use_gpu":use_gpu
}

waveform_class_kwargs = dict(inspiral_kwargs=inspiral_kwargs,
                              mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
                              sum_kwargs=sum_kwargs,
                              use_gpu=use_gpu)

waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(return_list=False)

if(use_gpu):
    xp=cp
else:
    xp=np

m1 = 1e6
m2 = 10
a = 0.8 # 0.95
e0 = 0.4 # 0.6 just spin first
xI0 = 1.0
dist = 0.4
qS = xp.pi/4
phiS = 1.0
qK = 1 
phiK = xp.pi/3
Phi_phi0 = 0.9
Phi_theta0 =0.5
Phi_r0 = 0.4

dt = 10.0
T = 0.2

chi2 = 0.0

dev_0_p=0.0
dev_0_e=0.0
dev_1_p=0.0
dev_1_e=0.0
dev_2_p=0.0
dev_2_e=0.0
evolve_1PA = False
evolve_primary = False
evolve_2PA = False
deviation_included=True
p0=7.5

print(use_gpu)
pars_list_com = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0,\
             chi2,evolve_1PA,evolve_primary,evolve_2PA,deviation_included,dev_0_p,dev_0_e,dev_1_p,dev_1_e,dev_2_p,dev_2_e]

add_param_args={"chi2":chi2,"evolve_1PA":evolve_1PA,"evolve_primary":evolve_primary,"evolve_2PA":evolve_2PA,"deviation_included":deviation_included,"dev0p":dev_0_p,
"dev0e":dev_0_e,"dev1p":dev_1_p,"dev1e":dev_1_e,"dev2p":dev_2_p,"dev2e":dev_2_e}

param_names = ['m1','m2','a','p0','e0','qS','phiS','Phi_phi0','Phi_r0','dev0p','dev0e']
add_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA,deviation_included,dev_0_p,dev_0_e,dev_1_p,dev_1_e,dev_2_p,dev_2_e]

emri_kwargs = {"T":T, "dt":dt}

param_names_com = ['m1','m2','a','p0','e0','xI0','dist','qS','phiS','qK','phiK','Phi_phi0','Phi_theta0','Phi_r0',"chi2",
               "evolve_1PA","evolve_primary","evolve_2PA","deviation_included","dev0p","dev0e","dev1p","dev1e","dev2p","dev2e"]
der_order = 8
Ndelta=10
sef = StableEMRIFisher(waveform_class=waveform_class, 
                       waveform_class_kwargs=waveform_class_kwargs,
                       waveform_generator=waveform_generator,
                       waveform_generator_kwargs=waveform_generator_kwargs,
                       stats_for_nerds = True, use_gpu = use_gpu,
                       deriv_type='stable',
                       noise_model=get_sensitivity,
                       noise_kwargs={'sens_fn':CornishLISASens,'return_type':'PSD'},
                       channels=["A","E"],
                       T = T, dt = dt,
                       stability_plot = False,
                       der_order = der_order, Ndelta = Ndelta,
                       plunge_check=True, return_derivatives=False
                       )
                   

SNR = sef.SNRcalc_SEF(*pars_list_com,**emri_kwargs,use_gpu=use_gpu)
print("SNR: ", SNR)

#initialize the 0PA (approximate) model
evolve_1PA = False 
evolve_primary = False
evolve_2PA = False #for approximate model
deviation_included=True
add_param_args={"chi2":chi2,"evolve_1PA":evolve_1PA,"evolve_primary":evolve_primary,"evolve_2PA":evolve_2PA,"deviation_included":deviation_included,"dev0p":dev_0_p,
"dev0e":dev_0_e,"dev1p":dev_1_p,"dev1e":dev_1_e,"dev2p":dev_2_p,"dev2e":dev_2_e}


param_names = ['m1','m2','a','p0','e0','qS','phiS','Phi_phi0','Phi_r0','dev0p','dev0e']
pars_list = [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0]

param_dict = {
    'm1': m1,
    'm2': m2,
    'a': a,
    'p0': p0,
    'e0': e0,
    'xI0': xI0,
    'dist': dist,
    'qS': qS,
    'phiS': phiS,
    'qK': qK,
    'phiK': phiK,
    'Phi_phi0': Phi_phi0,
    'Phi_theta0': Phi_theta0,
    'Phi_r0': Phi_r0
}


Fisher = sef(wave_params = param_dict,param_names=param_names, add_param_args=add_param_args,
            live_dangerously = False, stability_plot = True,der_order = der_order, Ndelta = Ndelta,
            )
def logmasstransform(Fisher, m1, index_of_m1 = 0):    
    J = np.eye(len(Fisher))
    J[index_of_m1,index_of_m1] = m1
    
    return J.T@Fisher@J
fisher_=logmasstransform(Fisher, m1, index_of_m1 = 0)
cov=np.linalg.inv(fisher_)
std= np.sqrt(np.diag(cov))
params_truth_in = np.array([np.log(m1), m2, a, p0, e0, qS, phiS, Phi_phi0,Phi_r0,dev_0_p,dev_0_e])
chi2=0
deviation_included=True
evolve_1PA=True
evolve_primary=False
evolve_2PA=False
use_gpu=True
add_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA,deviation_included,dev_0_p,dev_0_e,dev_1_p,dev_1_e,dev_2_p,dev_2_e]
superkludge_wave = GenerateEMRIWaveform(SuperKludgeWaveform,\
                                    sum_kwargs=sum_kwargs,\
                                    return_list=True,
                                    mode_selector_kwargs=dict(mode_selection_threshold=1e-5),
                                    inspiral_kwargs=inspiral_kwargs,
                                    use_gpu=use_gpu)
print(add_args)

waveform_true = superkludge_wave(m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK, Phi_phi0, Phi_theta0, Phi_r0, *add_args, dt=dt, T=T)
PSD=generate_PSD(waveform_true,dt,use_gpu=use_gpu,
                noise_PSD=get_sensitivity,
                noise_kwargs={'sens_fn':CornishLISASens,'return_type':'PSD'},
                channels=["A","E"])
chi2=0
deviation_included=True
evolve_1PA=False
evolve_primary=False
evolve_2PA=False
add_args = [chi2, evolve_1PA, evolve_primary, evolve_2PA,deviation_included,dev_0_p,dev_0_e,dev_1_p,dev_1_e,dev_2_p,dev_2_e]

def loglike_calc(m1_, m2_, a_, p0_, e0_,qS_,phiS_,Phi_phi0_,Phi_r0_,dev0p_,dev0e_):
    add_args__ = [chi2, evolve_1PA, evolve_primary, evolve_2PA,deviation_included,\
                dev0p_,dev0e_,dev_1_p,dev_1_e,dev_2_p,dev_2_e]
    waveform_temp=superkludge_wave(m1_, m2_, a_, p0_, e0_, xI0, dist, qS_, phiS_, qK, phiK, Phi_phi0_, Phi_theta0, Phi_r0_, *add_args__, dt=dt, T=T,use_gpu=use_gpu)
    diff_inner=inner_product(waveform_temp,waveform_true,PSD,dt,use_gpu=use_gpu)
    return -0.5 * diff_inner

def log_density(params):
    params = np.asarray(params)
    n_samples = params.shape[0] 
    log_likes = np.zeros(n_samples)
    for i in range(n_samples):
        logm1_, m2_, a_, p0_, e0_,qS_,phiS_,Phi_phi0_,Phi_r0_,dev0p_,dev0e_ = params[i]
        m1_ = 10**logm1_

        loglike = loglike_calc(m1_, m2_, a_, p0_, e0_,qS_,phiS_,Phi_phi0_,Phi_r0_,dev0p_,dev0e_)
        log_likes[i] = loglike 
    return log_likes

n=2
logm1lim = [max(0,params_truth_in[0] - n*std[0]), params_truth_in[0] + n*std[0]]
m2lim = [max(0,params_truth_in[1] - n*std[1]), params_truth_in[1] + n*std[1]]
alim = [max(-0.999,params_truth_in[2] - n*std[2]), min(params_truth_in[2] + n*std[2], 0.999)]  # a must be <1
p0lim = [max(0,params_truth_in[3] - n*std[3]), params_truth_in[3] + n*std[3]]
e0lim = [max(0,params_truth_in[4] - n*std[4]), min(1,params_truth_in[4] + n*std[4])]
qSlim = [params_truth_in[5] - n*std[5], params_truth_in[5] + n*std[5]]
phiSlim = [params_truth_in[6] - n*std[6], params_truth_in[6] + n*std[6]]
Phi_phi0lim = [params_truth_in[7] - n*std[7], params_truth_in[7] + n*std[7]]
Phi_r0lim = [params_truth_in[8] - n*std[8], params_truth_in[8] + n*std[8]]
dev0plim = [params_truth_in[9] - n*std[9], params_truth_in[9] + n*std[9]]
dev0elim = [params_truth_in[10] - n*std[10], params_truth_in[10] + n*std[10]]

def prior_transform(u):

    transformed = np.zeros_like(u)
    # Uniform in log for masses
    # m1
    transformed[:, 0] = (logm1lim[1] - logm1lim[0]) * u[:, 0] + logm1lim[0]

    # m2
    transformed[:, 1] = (m2lim[1] - m2lim[0]) * u[:, 1] + m2lim[0]

    # Linear in others 
    # a
    transformed[:, 2] = (alim[1] - alim[0]) * u[:, 2] + alim[0]
    transformed[:, 3] = (p0lim[1] - p0lim[0]) * u[:, 3] + p0lim[0] 
    transformed[:, 4] = (e0lim[1] - e0lim[0]) * u[:, 4] + e0lim[0]
    transformed[:, 5] = (qSlim[1] - qSlim[0]) * u[:, 5] + qSlim[0]
    transformed[:, 6] = (phiSlim[1] - phiSlim[0]) * u[:, 6] + phiSlim[0]
    transformed[:, 7] = (Phi_phi0lim[1] - Phi_phi0lim[0]) * u[:, 7] + Phi_phi0lim[0]
    transformed[:, 8] = (Phi_r0lim[1] - Phi_r0lim[0]) * u[:, 8] + Phi_r0lim[0]
    transformed[:, 9] = (dev0plim[1] - dev0plim[0]) * u[:, 9] + dev0plim[0]
    transformed[:, 10] = (dev0elim[1] - dev0elim[0]) * u[:, 10] + dev0elim[0]

    return transformed

    
def inverse_prior_transform(x):
    
    u = np.zeros_like(x)

    u[:, 0] = (x[:, 0] - logm1lim[0]) / (logm1lim[1] - logm1lim[0])
    u[:, 1] = (x[:, 1] - m2lim[0]) / (m2lim[1] - m2lim[0])
    u[:, 2] = (x[:, 2] - alim[0]) / (alim[1] - alim[0])
    u[:, 3] = (x[:, 3] - p0lim[0]) / (p0lim[1] - p0lim[0])
    u[:, 4] = (x[:, 4] - e0lim[0]) / (e0lim[1] - e0lim[0])
    u[:, 5] = (x[:, 5] - qSlim[0]) / (qSlim[1] - qSlim[0])
    u[:, 6] = (x[:, 6] - phiSlim[0]) / (phiSlim[1] - phiSlim[0])
    u[:, 7] = (x[:, 7] - Phi_phi0lim[0]) / (Phi_phi0lim[1] - Phi_phi0lim[0])
    u[:, 8] = (x[:, 8] - Phi_r0lim[0]) / (Phi_r0lim[1] - Phi_r0lim[0])
    u[:, 9] = (x[:, 9] - dev0plim[0]) / (dev0plim[1] - dev0plim[0])
    u[:, 10] = (x[:, 10] - dev0elim[0]) / (dev0elim[1] - dev0elim[0])
    
    return u
def analyze_results(sampler, savepath):
    """
    Analyze and summarize the sampling results.
    
    Parameters:
    ----------
    sampler : Sampler
        The sampler instance after running
    savepath : str
        Path where results are saved
    """
    print("\n" + "="*60)
    print("ANALYSIS RESULTS")
    print("="*60)
    
    # Get samples and weights
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    
    # Basic statistics
    print(f"Total samples generated: {len(samples):,}")
    print(f"Effective sample size: {1/np.sum(weights**2):.1f}")
    print(f"Weight coefficient of variation: {np.std(weights)/np.mean(weights):.3f}")
    
    # Transform samples to the coordinate system used in log_density
    transformed_samples = (samples * 1.4) - 0.2
    
    # Compute weighted statistics
    weighted_mean = np.average(transformed_samples, weights=weights, axis=0)
    weighted_std = np.sqrt(np.average((transformed_samples - weighted_mean)**2, 
                                    weights=weights, axis=0))
    
    print(f"\nWeighted mean (transformed coordinates): {weighted_mean}")
    print(f"Weighted std (transformed coordinates): {weighted_std}")
    
    # Find best samples
    log_densities = sampler.log_density_func(samples)
    best_idx = np.argmax(log_densities)
    best_sample = samples[best_idx]
    best_log_density = log_densities[best_idx]
    
    print(f"\nBest sample found:")
    print(f"  Coordinates (unit cube): {best_sample}")
    print(f"  Coordinates (transformed): {(best_sample * 1.4) - 0.2}")
    print(f"  Log-density: {best_log_density:.2f}")
    
    # Mode detection (simple clustering based on high-density samples)
    high_likelihood_threshold = np.percentile(log_densities, 95)
    high_likelihood_mask = log_densities >= high_likelihood_threshold
    high_likelihood_samples = transformed_samples[high_likelihood_mask]
    
    print(f"\nHigh-density regions (top 5%):")
    print(f"  Number of samples: {np.sum(high_likelihood_mask)}")
    
    if np.sum(high_likelihood_mask) > 0:
        # Simple mode detection: find cluster centers
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(10, np.sum(high_likelihood_mask))
            if n_clusters >= 2:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(high_likelihood_samples)
                cluster_centers = kmeans.cluster_centers_
                
                print(f"  Detected {n_clusters} high-density clusters:")
                for i, center in enumerate(cluster_centers):
                    cluster_size = np.sum(cluster_labels == i)
                    print(f"    Cluster {i+1}: center = {center}, samples = {cluster_size}")
        except ImportError:
            print("  (sklearn not available for clustering analysis)")
    
    # Save additional analysis results
    analysis_file = os.path.join(savepath, 'analysis_summary.txt')
    with open(analysis_file, 'w') as f:
        f.write(f"Multimodal Sampling Analysis Results\n")
        f.write(f"=====================================\n\n")
        f.write(f"Total samples: {len(samples)}\n")
        f.write(f"Effective sample size: {1/np.sum(weights**2):.1f}\n")
        f.write(f"Best log-density: {best_log_density:.6f}\n")
        f.write(f"Best sample (unit cube): {best_sample}\n")
        f.write(f"Best sample (transformed): {(best_sample * 1.4) - 0.2}\n")
    
    print(f"\nDetailed analysis saved to: {analysis_file}")

def visualize_marginal_distributions(sampler, savepath):
    """
    Create marginal distribution plots for each dimension.
    
    Parameters:
    ----------
    sampler : Sampler
        The sampler instance after running
    savepath : str
        Path where results are saved
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy.stats import norm
    except ImportError:
        print("Matplotlib/seaborn not available. Skipping visualization.")
        return
    
    print("\nCreating marginal distribution plots...")
    
    # Get samples and weights
    samples, weights = sampler.get_samples_with_weights(flatten=True)
    ndim = samples.shape[1]
    
    # Visualization parameters
    bin_num = 50
    decay = 3  # For exponential smoothing
    
def main():
    
    config = SamplerConfig(
        merge_confidence=0.9,          # Coverage prob → Mahalanobis merge radius R_m (higher is more permissive)
        alpha=10000,                    # Use recent samples for weighting
        trail_size=int(1e3),          # Maximum trials per iteration
        boundary_limiting=True,        # Enable boundary constraints
        use_beta=True,                # Use beta correction for boundaries
        integral_num=int(1e5),        # MC samples for beta estimation
        gamma=500,                    # Covariance update frequency
        exclude_scale_z=10,       # No exclusion based on weights
        use_pool=True,               # Set to True for multiprocessing
        n_pool=4                     # Number of processes (if use_pool=True)
    )
    
    ndim = 11
    n_seed = 100  # Number of initial processes
    init_cov_list = [np.eye(ndim) * 0.05] * n_seed
    sigma= 0.01

    savepath = './fisher_deviation_results/'

    init_cov_list = []
    for i in range(n_seed):
        init_cov_list.append(sigma**2 * np.eye(ndim))
    
    # Create save directory
    os.makedirs(savepath, exist_ok=True)
    
    print(f"Problem dimension: {ndim}")
    print(f"Number of processes: {n_seed}")
    print(f"Initial covariance scale: {sigma}")
    print(f"Save path: {savepath}")
    print(f"Multiprocessing: {config.use_pool}")

    # Initialize sampler
    print("\nInitializing sampler...")
    sampler = Sampler(
        ndim=ndim, 
        n_seed=n_seed,
        log_density_func=log_density,
        init_cov_list=init_cov_list,
        prior_transform=prior_transform,
        config=config
    )
   # Prepare initial samples using Latin Hypercube Sampling
    print("Preparing LHS samples...")
    sampler.prepare_lhs_samples(lhs_num=int(1e5), batch_size=100)
    
    # Run the sampling process
    print("Starting sampling process...")
    print("(This may take several minutes for 10,000 iterations)")
    
    try:
        sampler.run_sampling(
            num_iterations=10000, 
            savepath=savepath,
            print_iter=100  # Print progress every 100 iterations
        )
        
        print("\nSampling completed successfully!")
        
       # Analyze results
        analyze_results(sampler, savepath)
        
    #Create visualizations
        visualize_marginal_distributions(sampler, savepath)
        
    except KeyboardInterrupt:
        print("\nSampling interrupted by user.")
        print("Partial results have been saved.")
        if hasattr(sampler, 'searched_points_list'):
            analyze_results(sampler, savepath)
            try:
                visualize_marginal_distributions(sampler, savepath)
            except:
                print("Could not create visualizations with partial results.")
    
    except Exception as e:
        print(f"\nError during sampling: {e}")
        raise
    
    print(f"\nResults saved to: {savepath}")
    print("Example completed!")

if __name__ == "__main__":

    main()

