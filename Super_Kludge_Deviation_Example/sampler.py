from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer
from eryn.moves import GaussianMove, StretchMove, CombineMove
from eryn.utils.utility import groups_from_inds
import matplotlib.pyplot as plt
import numpy as np
#import cupy as cp
import likelihood_deviation
# set random seed
np.random.seed(53)
import corner
import sys
import likelihood_deviation
import os
import argparse



def parse_arguments(args=None):
    parser = argparse.ArgumentParser(description="To include output directory and data files")
    parser.add_argument('x_path', type=str)
    parser.add_argument('truth_path', type=str)
    parser.add_argument('cov_path', type=str)
    parser.add_argument('use_gpu', type=bool)
    parser.add_argument('output', type=str, help='output directory for corner plot')
    parser.add_argument('n_std', type=float, help='to set the prior range, n_std * std, give n_std')
    parser.add_argument('nwalkers', type=int, help='Number of Walkers')
    parser.add_argument('niter', type=int, help='Number of Waiterationlkers')
    args = parser.parse_args()
    return args

#    x=np.loadtxt(args.x_path)
#have to include ways which does not require truth and covaraince 
class Sampler(likelihood_deviation.EMRI_likelihood):
    def __init__(self, x, truth=None,cov=None,n_std=None,use_gpu=False):
        super().__init__(use_gpu,x)
        self.truth=truth
        self.cov=cov
        self.n_std=n_std

        #Define some default parameters
        #have to make this thing better for future use
        self.set_args(xI0=1.0,T=0.1,dt=10.0,evolve_1PA = False,evolve_primary = False,evolve_2PA = False,
                  deviation_included=False,qK = 1,phiK = 1 + np.pi/3,Phi_theta0 =0.2,chi2=0.8,dev_0=[0,0],
                  dev_1=[0,0],dev_2=[0,0])
       # self.param_names = ['m1','m2','a','p0','e0','dist','qS','phiS','Phi_phi0','Phi_r0']

    def log_like_likelihood(self,x_param):
        print(x_param)
        self.set_args_faster_likelihood(m1=10**x_param[0],m2=10**x_param[1],a=x_param[2],p0=x_param[3],
                      e_0=x_param[4],dist=x_param[5],qS=x_param[6],phiS=x_param[7],Phi_phi0=x_param[8],Phi_r0=x_param[9])
        try:
            log_like = self.likelihood()
        except:
            print(log_like)
            print("Exception Occured")
            return -np.inf      # a very high value 
        return log_like
    
    def compute_ranges(self):
        N = self.truth.size
        #first to check if covariance matrix makes sense
        if self.cov.ndim == 1:
            if self.cov.size != N:
                raise ValueError(f"Variance vector length ({self.cov.size}) doesn't match truth length ({N}).")
            else:   #maybe it's just the variance matrix
                variance=self.cov
        elif self.cov.ndim == 2:
            if self.cov.shape[0] != self.cov.shape[1]:
                raise ValueError("Covariance input is 2D but not square. Are you sure it is the covariance matrix")
            if self.cov.shape[0] != N:
                raise ValueError(f"Covariance matrix size ({self.cov.shape[0]}) doesn't match truth length ({N}).")
            variances = np.diag(self.cov)
        else:
            raise ValueError("Covariance must be either 1D variances or a 2D covariance matrix.")

        neg = variances <= 0
        if np.any(neg):
                raise ValueError(f"Covariance diagonal contains negative values or zero: {variances[neg]}")
        std = np.sqrt(variances)
        lower = self.truth - self.n_std * std
        upper = self.truth + self.n_std * std
        
        lower[0:2] = np.log10(lower[0:2])
        upper[0:2] = np.log10(upper[0:2])

        ranges = np.vstack([lower, upper]).T  # shape (N,2)
        print("ranges\n",ranges)
       # ranges_list = [tuple(r) for r in ranges.tolist()]
        return ranges

if __name__ == "__main__":
    args=parse_arguments() #see it works with CLI also
    nwalkers = args.nwalkers
    truth=np.asarray(np.loadtxt(args.truth_path))
    Cov_matrix=np.asarray(np.loadtxt(args.cov_path))
    x=np.loadtxt(args.x_path)
    ndim = len(truth)
    # print(Cov_matrix)
    # print(Cov_matrix.shape)
    sampler=Sampler(x,truth,Cov_matrix,args.n_std,args.use_gpu)

    try:
        ranges = sampler.compute_ranges()
    except ValueError as e:
        print("Error computing ranges:", e, file=sys.stderr)
        sys.exit(2)
  
    priors_in = {i: uniform_dist(ranges[i][0], ranges[i][1]) for i in range(ndim)}
    priors = ProbDistContainer(priors_in)
    check_truth=truth
    check_truth[0]=np.log10(truth[0])
    check_truth[1]=np.log10(truth[1])
    print("Shoud be zero",sampler.log_like_likelihood(check_truth))
    ensemble = EnsembleSampler(
        nwalkers,
        ndim,
        sampler.log_like_likelihood,
        priors)
    # starting positions randomized throughout prior
    coords = priors.rvs(size=(nwalkers,))
    # check log_like
    log_like_test = np.asarray([sampler.log_like_likelihood(coords[i]) for i in range(nwalkers)])
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
    fig.savefig(output_+"/try_first_sampling.png", dpi=300, bbox_inches="tight")
    ll = ensemble.backend.get_log_like()
    lp = ensemble.backend.get_log_prior()
    print(f"Number of iterations {ensemble.backend.iteration}\n")
    # equivalent to ensemble.get_log_like() and ensemble.get_log_prior()
    print(ll.shape, ll, lp)