import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import Optional, Type, Union
from scipy.interpolate import CubicSpline
import time

#few imports
from few.trajectory.inspiral import EMRIInspiral   #function to generate trajectories
from few.trajectory.ode.base import ODEBase        #superclass for defining modified flux trajectories
from few.trajectory.ode.flux import KerrEccEqFlux  #Adiabatic KerrEccEq base trajectory class

from few.amplitude.ampinterp2d import AmpInterpKerrEccEq
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.modeselector import ModeSelector
from few.utils.exceptions import TrajectoryOffGridException

from few.waveform import (
    GenerateEMRIWaveform
)

##### SuperKludge Imports #####
from few.trajectory.ode.flux import SuperKludgeFlux
from few.waveform.waveform import SuperKludgeWaveform #waveform module

from few.waveform.base import SphericalHarmonicWaveformBase #generic waveform generation class
from few.utils.baseclasses import SphericalHarmonic, KerrEccentricEquatorial, ParallelModuleBase, BackendLike

from few.utils.constants import YRSID_SI
from few.utils.utility import ( 
    get_mismatch, 
    get_m2_at_t, 
    get_p_at_t, 
    )
from few.utils.geodesic import (
    get_fundamental_frequencies,
    get_separatrix,
    get_kerr_geo_constants_of_motion,
    ELQ_to_pex,
    )

#class EMRI_likelihood(SuperKludgeFlux,SuperKludgeWaveform):
class EMRI_likelihood():
    "likelihood class for our sampler"
    def __init__(self,x,m1,m2,a,e_0,Y0,dist,qs,phiS,qK,phiK,Phi_phi_0,phi_theta0,phi_r0,dt,T,chi2,evolve_1PA\
                 ,evolve_primary,evolve_2PA,deviation_included,dev_0,dev_1,dev_2):
        self.x=x
        self.m1 = m1
        self.m2 = m2
        self.a =  a
        self.e_0 = e_0
        self.Y0 = Y0
        self.dist = dist
        self.qS = qs
        self.phiS = phiS
        self.qK = qK
        self.phiK = phiK
        self.Phi_phi_0 = Phi_phi_0
        self.phi_theta0 = phi_theta0
        self.Phi_r0 = phi_r0
        self.dt = dt #in seconds
        self.T = T  #in years
        self.chi2 = chi2
        self.evolve_1PA =evolve_1PA
        self.evolve_primary =evolve_primary
        self.evolve_2PA = evolve_2PA
        self.deviation_included = deviation_included
        self.dev_0_p= dev_0[0]
        self.dev_0_e= dev_0[1]
        self.dev_1_p= dev_1[0]
        self.dev_1_e= dev_1[1]
        self.dev_2_p= dev_2[0]
        self.dev_2_e= dev_2[1]
    
    def set_args(self):
        self.add_args=[self.chi2, self.evolve_1PA, self.evolve_primary, self.evolve_2PA,self.deviation_included,\
                    self.dev_0_p,self.dev_0_e,self.dev_1_p,self.dev_1_e,self.dev_2_p,self.dev_2_e]
    

    def inspiral_evolution(self):

        #shubham's note: #get_p_at_t may fail with SuperKludge, especially with lower eccentricities, so we use KerrEccEq to generate p0 

        SK_traj = EMRIInspiral(func=SuperKludgeFlux)
        p0 = get_p_at_t(traj_module=SK_traj, t_out=self.T, traj_args=[self.m1, self.m2, self.a, self.e0, self.Y0, *self.add_args]) + 0.1
        self.p0=p0
    

#FFT algorithms are most efficient (and sometimes require) input lengths that are powers of 2.
    def zero_pad(data):
        N = len(data)
        pow_2 = np.ceil(np.log2(N))
        return np.pad(data,(0,int((2**pow_2)-N)),'constant')
    
    def P_OMS(f):
        return (1.5e-11)**2 * (1 + (2e-3/f)**4)
    
    def P_acc(f):
        return (3e-15)*(1+((4e-4/f)**2))*(1+(f/8e-3)**4)
    
    def confusion_noise(f):
        #depends on observation period
        return 0.0
    
    #used from mismatch.py
    def FFT(data):
        data_pad = zero_pad(data)
        data_f = np.fft.rfft(data_pad)[1:]
        return data_f
    
    def get_psd(f):
        psd=(10/(3*(2.5e9**2))) * (P_OMS(f) + 2*(1+np.pow(np.cos(f/19.09e-3),2)) * P_acc(f)/((2*np.pi*f)**4))*(1+ 0.6 * (f/19.09e-3)**2)
    
        #used from mismatch.py

    def inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD):
        prefac = 4*delta_t / N_t
        sig2_f_conj = np.conjugate(sig2_f)
        return prefac * np.real(np.sum((sig1_f * sig2_f_conj)/PSD))

    def overlap_f(sig1_f,sig2_f,N_t,delta_t,PSD):
        aa = inner_prod(sig1_f,sig1_f,N_t,delta_t,PSD)
        bb = inner_prod(sig2_f,sig2_f,N_t,delta_t,PSD)
        ab = inner_prod(sig1_f,sig2_f,N_t,delta_t,PSD)
        return  (ab/(cp.sqrt(bb*aa)))
    
    


    def waveform_generation(self,use_gpu=False):
        use_gpu = use_gpu
        self.inspiral_evolution(self)

        #have to see if we need all these calculated or not
        # keyword arguments for inspiral generator (EMRIInspiral)
        inspiral_kwargs={
                "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
                "buffer_length": int(1e3),  # all of the trajectories will be well under len = 1000
            }

        # keyword arguments for inspiral generator (RomanAmplitude)
        amplitude_kwargs = {
            "buffer_length": int(1e3),  # all of the trajectories will be well under len = 1000
            "use_gpu": use_gpu  # GPU is available in this class
        }

        # keyword arguments for Ylm generator (GetYlms)
        Ylm_kwargs = {
            "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
        }

        # keyword arguments for summation generator (InterpolatedModeSum)
        sum_kwargs = {
            "use_gpu": use_gpu,  # GPU is available for this type of summation
            "pad_output": False,
        }

        #time or freq?
        superkludge_wave = GenerateEMRIWaveform(SuperKludgeWaveform,
                                    inspiral_kwargs=inspiral_kwargs,
                                    amplitude_kwargs=amplitude_kwargs,
                                    Ylm_kwargs=Ylm_kwargs,
                                    sum_kwargs=sum_kwargs,
                                    # output_type=return_type,
                                    use_gpu=use_gpu,
                                    return_list=True)

        waveform = superkludge_wave(self.m1, self.m2, self.a, self.p0, self.e_0, self.Y0, self.dist, self.qS,\
                                     self.phiS, self.qK, self.phiK, self.Phi_phi_0, self.phi_theta0, self.Phi_r0,\
                                       *self.add_args, dt=self.dt, T=self.T)
      #from waveform tutorial

        fft_TD_plus = np.fft.fftshift(np.fft.fft(waveform[0])) * self.dt
        freq_plus = np.fft.fftshift(np.fft.fftfreq(len(waveform[0]), self.dt))
        # define the positive frequencies
        positive_frequency_mask_plus = freq_plus >= 0.0

        fft_TD_x = np.fft.fftshift(np.fft.fft(self.x[0])) * self.dt
        freq_x = np.fft.fftshift(np.fft.fftfreq(len(self.x[0]), self.dt))
        positive_frequency_x = freq_x >= 0.0

        psd_plus = psd(freq_plus[positive_frequency_mask_plus]) 
        N_t = 2**np.ceil(np.log2(len(waveform[0].real)))
        plus_ = inner_prod(fft_TD_plus[positive_frequency_mask_plus], fft_TD_plus[positive_frequency_mask_plus], psd_plus)
        
 

# # mismatch
# psd = get_sensitivity(freq[positive_frequency_mask]) / np.diff(freq)[0]
# t
# )
# fd_fd = inner_product(hf[0], hf[0], psd)
# Mism = np.abs(
#     1
#     - inner_product(fft_TD[positive_frequency_mask], hf[0], psd)
#     / np.sqrt(td_td * fd_fd)
# )
# print("mismatch", Mism)
# # SNR
# print("TD SNR", np.sqrt(td_td))
# print("FD SNR", np.sqrt(fd_fd))
        

        





