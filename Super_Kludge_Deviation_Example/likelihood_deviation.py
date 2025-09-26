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
#FFT algorithms are most efficient (and sometimes require) input lengths that are powers of 2.But it is not used in this code
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

def get_psd(f):
    psd=(10/(3*(2.5e9**2))) * (P_OMS(f) + 2*(1+np.pow(np.cos(f/19.09e-3),2)) * P_acc(f)/((2*np.pi*f)**4))*(1+ 0.6 * (f/19.09e-3)**2)

    #used from mismatch.py
def inner_product(x, h, del_f,sn_f): 
    return 4 * del_f* np.real(np.sum(np.conj(x) * h / sn_f))

#class EMRI_likelihood(SuperKludgeFlux,SuperKludgeWaveform):
class EMRI_likelihood():
    "likelihood class for our sampler"
    def __init__(self,use_gpu,x):
        self.use_gpu=use_gpu

        sum_kwargs = {"pad_output": False}

        #time or freq?
        superkludge_wave = GenerateEMRIWaveform(SuperKludgeWaveform,
                                    sum_kwargs=sum_kwargs,
                                    # output_type=return_type,
                                    use_gpu=use_gpu,
                                    return_list=True)
        
        SK_traj = EMRIInspiral(func=SuperKludgeFlux)
        self.superkludge_wave=superkludge_wave
        self.SK_traj=SK_traj
        self.x=x
        
    def set_args(self,m1,m2,a,e_0,Y0,dist,qs,phiS,qK,phiK,Phi_phi_0,phi_theta0,phi_r0,dt,T,chi2,evolve_1PA\
                 ,evolve_primary,evolve_2PA,deviation_included,dev_0,dev_1,dev_2):
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


    def inspiral_evolution(self):

        #shubham's note: #get_p_at_t may fail with SuperKludge, especially with lower eccentricities, so we use KerrEccEq to generate p0 
        p0 = get_p_at_t(traj_module=self.SK_traj, t_out=self.T, traj_args=[self.m1, self.m2, self.a, self.e0, self.Y0, *self.add_args]) + 0.1
        self.p0=p0

    def likelihood(self):
        
        #use set_args before running it
        self.inspiral_evolution(self)
        waveform = self.superkludge_wave(self.m1, self.m2, self.a, self.p0, self.e_0, self.Y0, self.dist, self.qS,\
                                     self.phiS, self.qK, self.phiK, self.Phi_phi_0, self.phi_theta0, self.Phi_r0,\
                                       *self.add_args, dt=self.dt, T=self.T)
      #from waveform tutorial and just taking h0
        N=len(self.x[0])
        # fft_TD_x = np.fft.fftshift(np.fft.fft(self.x[0])) * self.dt
        # fft_TD_plus = np.fft.fftshift(np.fft.fft(waveform[0])) * self.dt

        # freq_x = np.fft.fftshift(np.fft.fftfreq(N, self.dt))
        # positive_frequency_x = freq_x > 0.0
        # #greather than zero or greater than equal to

        # fft_TD_plus_pos = fft_TD_plus[positive_frequency_x]
        # fft_TD_x_pos = fft_TD_x[positive_frequency_x]
        # psd_plus = get_psd(fft_TD_x_pos) 
        # x_h = inner_product(fft_TD_plus_pos, fft_TD_x_pos, np.diff(freq_x)[0],psd_plus)
        x_diff_h=self.x[0]-waveform[0]
        fft_x_diff_h = np.fft.fftshift(np.fft.fft(x_diff_h)) * self.dt
        freq_x_diff_h = np.fft.fftshift(np.fft.fftfreq(N, self.dt))
        positive_freq_marker = freq_x_diff_h > 0.0
        pos_freq=freq_x_diff_h[positive_freq_marker]

        fft_TD_freq_x_diff_h_pos = fft_x_diff_h[positive_freq_marker]
        psd_plus = get_psd(pos_freq) 

        x_h = inner_product(fft_TD_freq_x_diff_h_pos, fft_TD_freq_x_diff_h_pos, np.diff(freq_x_diff_h)[0],psd_plus)
        likelihood=-0.5 * x_h

        return likelihood
