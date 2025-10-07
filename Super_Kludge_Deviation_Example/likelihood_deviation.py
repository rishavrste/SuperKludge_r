#chnage psd calculation
import numpy as np
# import cupy as cp
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
    return psd

    #used from mismatch.py
def inner_product(x, h, del_f,sn_f): 
    first=4 * del_f* np.real(np.sum(np.conj(x[:,0]) * h[:,0] / sn_f))
    return first + 4 * del_f* np.real(np.sum(np.conj(x[:,1]) * h[:,1] / sn_f))

#class EMRI_likelihood(SuperKludgeFlux,SuperKludgeWaveform):
class EMRI_likelihood():
    "likelihood class for our sampler"
    def __init__(self,use_gpu:bool,x):
        self.use_gpu=use_gpu

        sum_kwargs = {"pad_output": True}

        #time or freq?
        superkludge_wave = GenerateEMRIWaveform(SuperKludgeWaveform,
                                    sum_kwargs=sum_kwargs,
                                    #output_type=return_type,
                                    use_gpu=use_gpu,
                                    return_list=True)
        
        SK_traj = EMRIInspiral(func=SuperKludgeFlux)
        self.superkludge_wave=superkludge_wave
        self.SK_traj=SK_traj
        self.x=x
        self.N=len(self.x[0])
        
    def set_args(self,m1=None, m2=None, a=None, e_0=None, p0=None, xI0=None, dist=None,qS=None, phiS=None,
                  qK=None, phiK=None,Phi_phi0=None, Phi_theta0=None, Phi_r0=None,dt=None, T=None, chi2=None,
                  evolve_1PA=None, evolve_primary=None, evolve_2PA=None,
                  deviation_included=None,dev_0=None, dev_1=None, dev_2=None):
          # Only update the ones that are given
        if m1 is not None: self.m1 = m1
        if m2 is not None: self.m2 = m2
        if a is not None: self.a = a
        if e_0 is not None: self.e_0 = e_0
        if p0 is not None: self.p0 = p0
        if xI0 is not None: self.xI0 = xI0
        if dist is not None: self.dist = dist
        if qS is not None: self.qS = qS
        if phiS is not None: self.phiS = phiS
        if qK is not None: self.qK = qK
        if phiK is not None: self.phiK = phiK
        if Phi_phi0 is not None: self.Phi_phi_0 = Phi_phi0
        if Phi_theta0 is not None: self.phi_theta0 = Phi_theta0
        if Phi_r0 is not None: self.Phi_r0 = Phi_r0
        if dt is not None: self.dt = dt
        if T is not None: self.T = T
        if chi2 is not None: self.chi2 = chi2
        if evolve_1PA is not None: self.evolve_1PA = evolve_1PA
        if evolve_primary is not None: self.evolve_primary = evolve_primary
        if evolve_2PA is not None: self.evolve_2PA = evolve_2PA
        if deviation_included is not None: self.deviation_included = deviation_included
        if dev_0 is not None:
            self.dev_0_p, self.dev_0_e = dev_0
        if dev_1 is not None:
            self.dev_1_p, self.dev_1_e = dev_1
        if dev_2 is not None:
            self.dev_2_p, self.dev_2_e = dev_2
        self.add_args=[self.chi2, self.evolve_1PA, self.evolve_primary, self.evolve_2PA,self.deviation_included,self.dev_0_p,
                       self.dev_0_e,self.dev_1_p,self.dev_1_e,self.dev_2_p,self.dev_2_e]
#if else might be time extensive
    def set_args_faster_likelihood(self,m1,m2,a,p0,e_0,dist,qS,phiS,Phi_phi0,Phi_r0):
        self.m1=m1
        self.m2=m2
        self.a=a
        self.p0=p0
        self.e_0=e_0
        self.dist=dist
        self.qS=qS
        self.phiS=phiS
        self.Phi_phi_0=Phi_phi0
        self.Phi_r0=Phi_r0
            
    def inspiral_evolution(self):
        #shubham's note: #get_p_at_t may fail with SuperKludge, especially with lower eccentricities, so we use KerrEccEq to generate p0 
        p0 = get_p_at_t(traj_module=self.SK_traj, t_out=self.T, traj_args=[self.m1, self.m2, self.a, self.e0, self.xI0, *self.add_args]) + 0.1
        self.p0=p0

    def likelihood(self):
        
        #self.inspiral_evolution(self)
        #print("Trying waveform genreation, does the errro occur here")
        #print(self.superkludge_wave)
        #start=time.time()
        waveform = self.superkludge_wave(self.m1, self.m2, self.a, self.p0, self.e_0, self.xI0, self.dist, self.qS,\
                                     self.phiS, self.qK, self.phiK, self.Phi_phi_0, self.phi_theta0, self.Phi_r0,\
                                       *self.add_args, dt=self.dt, T=self.T)
        #print(time.time()-start)
      #from waveform tutorial and just taking h0
        
        # fft_TD_x = np.fft.fftshift(np.fft.fft(self.x[0])) * self.dt
        # fft_TD_plus = np.fft.fftshift(np.fft.fft(waveform[0])) * self.dt

        # freq_x = np.fft.fftshift(np.fft.fftfreq(N, self.dt))
        # positive_frequency_x = freq_x > 0.0
        # #greather than zero or greater than equal to

        # fft_TD_plus_pos = fft_TD_plus[positive_frequency_x]
        # fft_TD_x_pos = fft_TD_x[positive_frequency_x]
        # psd_plus = get_psd(fft_TD_x_pos) 
        # x_h = inner_product(fft_TD_plus_pos, fft_TD_x_pos, np.diff(freq_x)[0],psd_plus)
        x_diff_h=self.x-waveform
        fft_x_diff_h = np.fft.fftshift(np.fft.fft(x_diff_h)) * self.dt
        freq_x_diff_h = np.fft.fftshift(np.fft.fftfreq(self.N, self.dt))
        positive_freq_marker = freq_x_diff_h > 0.0
        pos_freq=freq_x_diff_h[positive_freq_marker]

        fft_TD_freq_x_diff_h_pos_1 = fft_x_diff_h[0][positive_freq_marker]
        fft_TD_freq_x_diff_h_pos_2 = fft_x_diff_h[1][positive_freq_marker]
        fft_TD_freq_x_diff_h_pos = np.stack((fft_TD_freq_x_diff_h_pos_1, fft_TD_freq_x_diff_h_pos_2), axis=1)
        psd_plus = get_psd(pos_freq)
        # print(len(psd_plus))
        # print(fft_TD_freq_x_diff_h_pos.shape)
        likelihood = -0.5 * inner_product(fft_TD_freq_x_diff_h_pos, fft_TD_freq_x_diff_h_pos, np.diff(freq_x_diff_h)[0],psd_plus)
        print(likelihood)
        return likelihood
