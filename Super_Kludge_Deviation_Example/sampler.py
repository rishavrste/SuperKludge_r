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
import likelihood_deviation
import os
import argparse

parser = argparse.ArgumentParser(description="To include output directory and true data")

parser.add_argument('output', type=float, help='output directory for corner plot')
parser.add_argument('x_path', type=str)

args = parser.parse_args()
x=np.loadtxt(args.x_path)

emri=likelihood_deviation.EMRI_likelihood(False)
 

def log_like_likelihood(x):
    X=x+[] #whatever element we need to be constant
    emri.set_args(X)
    try:
        log_like = emri.likelihood()
    except:
        return -np.inf # a very high value 
    return log_like


ndim = 10 # or so
nwalkers = 500

truth=[]
range=[[]]  #bound value
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


nsteps = 5000
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