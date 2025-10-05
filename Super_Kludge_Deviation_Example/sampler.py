from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds
import matplotlib.pyplot as plt
import numpy as np

# set random seed
np.random.seed(42)
import corner
import sys
import likelihood_deviation
import os
import argparse


def compute_ranges(truth, cov, n_std=5):

    N = truth.size
    #first to check if covariance matrix makes sense
    if cov.ndim == 1:
        if cov.size != N:
            raise ValueError(f"Variance vector length ({cov.size}) doesn't match truth length ({N}).")
        else:   #maybe it's just the variance matrix
            variance=cov
    elif cov.ndim == 2:
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance input is 2D but not square. Are you sure it is the covariance matrix")
        if cov.shape[0] != N:
            raise ValueError(f"Covariance matrix size ({cov.shape[0]}) doesn't match truth length ({N}).")
        variances = np.diag(cov)
    else:
        raise ValueError("Covariance must be either 1D variances or a 2D covariance matrix.")
    
    small_neg = variances <= 0
    if np.any(small_neg):
            raise ValueError(f"Covariance diagonal contains negative values or zero: {variances[small_neg]}")

    std = np.sqrt(variances)
    lower = truth - n_std * std
    upper = truth + n_std * std
    ranges = np.vstack([lower, upper]).T  # shape (N,2)
    ranges_list = [tuple(r) for r in ranges.tolist()]
    return ranges, ranges_list

parser = argparse.ArgumentParser(description="To include output directory and true data")
parser.add_argument('x_path', type=str)
parser.add_argument('truth_path', type=str)
parser.add_argument('cov_path', type=str)
parser.add_argument('use_gpu', type=str)
parser.add_argument('output', type=float, help='output directory for corner plot')
args = parser.parse_args()
x=np.loadtxt(args.x_path)

#Define some default parameters
emri=likelihood_deviation.EMRI_likelihood(args.use_gpu,x)
emri.set_args(Y0=1.0,T=0.5,dt=10.0,evolve_1PA = False,evolve_primary = False,evolve_2PA = False,
              deviation_included=False,qK = 1,phiK = 1 + np.pi/3,Phi_theta0 =0.2)


def log_like_likelihood(x):
    param_names = ['m1','m2','p0','e0','dist','qS','phiS','Phi_phi0','Phi_r0','chi2'] #works
    emri.set_args(m1=10**x[0],m2=10**x[1],p0=x[2],e0=x[3],dist=x[4],qS=x[5],phiS=x[6],Phi_phi0=x[7],phi_r0=x[8],chi2=x[9])
    try:
        log_like = emri.likelihood()
    except:
        print("Exception Occured")
        return -np.inf      # a very high value 
    return log_like


ndim = 10 # or so
nwalkers = 500

truth=np.asarray(np.loadtxt(args.x_path))
Cov_matrix=np.asarray(np.loadtxt(args.cov_path))

try:
    ranges, ranges_list = compute_ranges(truth, Cov_matrix, n_std=args.n_std)
except ValueError as e:
    print("Error computing ranges:", e, file=sys.stderr)
    sys.exit(2)

priors_in = {i: uniform_dist(range[i][0], range[i][1]) for i in range(ndim)}
priors = ProbDistContainer(priors_in)

ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    log_like_likelihood,
    priors)

# starting positions randomized throughout prior
coords = priors.rvs(size=(nwalkers,))

# check log_like
log_like_test = np.asarray([log_like_likelihood(coords[i]) for i in range(nwalkers)])
if np.any(log_like_test == -np.inf):
    print("Check your prior or likelihood function")


# check log_prior
log_prior_test = np.asarray([priors.logpdf(coords[i]) for i in range(nwalkers)])
print("\nLog-prior - should be constant for uniform prior for a paramter:\n", log_prior_test)


nsteps = 20000
burn = 250
thin_by = 50
out = ensemble.run_mcmc(coords, nsteps, burn=burn, progress=True, thin_by=thin_by)

samples = ensemble.get_chain()['model_0'].reshape(-1, ndim)
fig = corner.corner(samples, truths=truth)

output_=args.output
os.makedirs(output_, exist_ok=True)
fig.savefig(output_+"/try.png", dpi=300, bbox_inches="tight")

ll = ensemble.backend.get_log_like()
lp = ensemble.backend.get_log_prior()

print(f"Number of iterations {ensemble.backend.iteration}\n")
# equivalent to ensemble.get_log_like() and ensemble.get_log_prior()
print(ll.shape, ll, lp)