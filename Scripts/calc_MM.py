import lalsimulation
import lal
import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from pycbc.filter.matchedfilter import optimized_match
from scipy.optimize import minimize
import matplotlib.tri as tri
print('Modules Loaded')
def easy_match(fs,h1,h2,psd_function,zero_pad_factor=3):
    '''
        Light weight wrapper to calculate the match between
        two FD waveforms h1 and h2. See match calculation in Mathematica.

        Agrees with pycbc match function.
    '''
    if(len(h1) != len(h2)):
        raise Exception('Error: length of h1 and h2 do not match!')
    #if not(is_uniformly_sampled(fs)):
    #    raise Exception('Error: frequencies are not uniformly sampled.')

    length       = len(h1)

    # Here, psd_function(100) multiplies everything by some term \alpha
    # which cancels at the end. This is just used for data conditioning.
    Sn_array     = psd_function(100) / np.array( list( map(psd_function, fs) ) )
    Sn_array[-1] = Sn_array[-2]

    Ah1          = np.abs(h1)
    Ah2          = np.abs(h2)
    norm1        = np.dot(Ah1, Ah1 * Sn_array)
    norm2        = np.dot(Ah2, Ah2 * Sn_array)

    # Get the integrand
    integrand    = h1 * h2.conj() * Sn_array

    # Zero pad the integrand
    integrand_zp = np.pad(integrand,(0,zero_pad_factor * length),'constant',constant_values=(0,0))

    # Get the complex snr
    complex_snr  = np.asarray( np.fft.fft(integrand_zp) )
    match        = np.max(np.abs(complex_snr)) / np.sqrt(norm1 * norm2)

    return match


def get_mismatch(f, h1, h2, flow, fhigh, method = None):
    idx = (f>flow) * (f<fhigh)
    # apply band
    _f = f[idx]
    h1_norm = h1[idx]
    h2_norm = h2[idx]
    # normalize h1 and h2
    h1_norm /= np.sqrt(dot(_f, h1_norm, h1_norm))
    h2_norm /= np.sqrt(dot(_f, h2_norm, h2_norm))
    # maximize overlap by linear offset in f
    if method == None:
      method = ["Nelder-Mead", "Powell", "CG", "BFGS", "TNC", "L-BFGS-B", "COBYLA", "SLSQP"]
      #method = ["Nelder-Mead", "Powell", "CG", "BFGS"]
    else:
        method = [method]

    def _mismatch(c):
        mismatch = 1.-dot(_f, h1_norm, h2_norm*np.exp(+1j*(c[0]+c[1]*2.*np.pi*_f)))
        return mismatch

    minmismatch = 1.0
    count = 5
    counter = 0
    cguess  = np.array([0.,0.])    
    #off = [0,0.0001,0.002,0.005,0.005,0.1,0.5,1.0,2.,3.,10]

    while minmismatch > 1e-2 and counter <= count:
        result_mismatch = []
        for imethod in method:
            res = minimize(_mismatch, cguess, method = imethod)
            result_mismatch.append(_mismatch(res.x))
        cguess = cguess + 0.02
        #print(cguess)
        minmismatch = np.min(result_mismatch)
        counter = counter + 1.  
        #print(minmismatch)

    indexmin = result_mismatch.index(minmismatch)
    method_used = method[indexmin]
    return minmismatch, method_used

def dot(f, h1, h2):
    df = f[1]-f[0]
    result = np.real(4.*df*np.nansum(np.conjugate(h1)*h2/1))
    return result

def get_quadrupole(EOS, m, chi):
    c = 299792458.
    G = 6.6743015e-11
    Msun = 1.9891e30

    GR_CM2_fac = 1e-7
    RNS_Q_FAC = 1e42
    GEOM_FAC = c**4/G**2
    if EOS =='stiff':
        Q = 568.011*m*chi**2 - 335.63*chi**3 + 62.3359*m**2*chi**3
    elif EOS == 'soft':
        Q = 507.293*m*chi**2 - 44.8039*m**3*chi**2 - 378.495*chi**3 + 315.377*m*chi**3
    return Q*RNS_Q_FAC*GR_CM2_fac*GEOM_FAC/(m*Msun)**3/(chi**2)

approx = lalsimulation.IMRPhenomXAS_NRTidalv3
incl = np.pi
srate = 4096
deltaF = 1/srate
fmin = 5.
fref = fmin
fmax = srate/2

PSD = ius(np.arange(0, fmax,deltaF), np.ones(len(np.arange(0, fmax,deltaF))))

eos_data = np.loadtxt('/work/williams5/testing-urs/EOS/soft_mrlfmode.dat')
eos_mass = eos_data[:,1]
eos_lambda = eos_data[:,2]

soft_ml_interp = ius(eos_mass, eos_lambda)

eos_data = np.loadtxt('/work/williams5/testing-urs/EOS/stiff_mrlfmode.dat')
eos_mass = eos_data[:,1]
eos_lambda = eos_data[:,2]

stiff_ml_interp = ius(eos_mass, eos_lambda)

n=50
masses = np.linspace(1.0,2.2,n)
m1s, m2s = np.meshgrid(masses,masses)

#print('Binary Love')
#print('Soft')
#MM = np.zeros((n,n))

#k=0
#for j, m1 in enumerate(masses):
#    for i, m2 in enumerate(masses):
#        if m2>m1:
#            MM[i,j]= None
#            continue
#        L1 = soft_ml_interp(m1).item()
#        L2 = soft_ml_interp(m2).item()
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_true = hptilde.data.data - 1j*hctilde.data.data
        
#        Ls = 0.5*(L1+L2)
#        La_UR = bilby.gw.conversion.binary_love_fit_lambda_symmetric_mass_ratio_to_lambda_antisymmetric(Ls, m2/m1)  
#        L1_UR, L2_UR = bilby.gw.conversion.lambda_symmetric_lambda_antisymmetric_to_lambda_1_lambda_2(Ls, La_UR)
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1_UR)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2_UR)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_UR = hptilde.data.data - 1j*hctilde.data.data
#        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
#        MM[i,j] = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
#        k+=1
#        print(k)
#np.savetxt('./soft_MM_2048.txt', MM)

#print('Stiff')

#MM_stiff = np.zeros((n,n))
#k=0
#for j, m1 in enumerate(masses):
#    for i, m2 in enumerate(masses):
#        if m2>m1:
#            MM_stiff[i,j]= None
#            continue
#        L1 = stiff_ml_interp(m1).item()
#        L2 = stiff_ml_interp(m2).item()
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_true = hptilde.data.data - 1j*hctilde.data.data
#        
#        Ls = 0.5*(L1+L2)
#        La_UR = bilby.gw.conversion.binary_love_fit_lambda_symmetric_mass_ratio_to_lambda_antisymmetric(Ls, m2/m1)  
#        L1_UR, L2_UR = bilby.gw.conversion.lambda_symmetric_lambda_antisymmetric_to_lambda_1_lambda_2(Ls, La_UR)
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1_UR)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2_UR)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_UR = hptilde.data.data - 1j*hctilde.data.data
#        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
#        MM_stiff[i,j] = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
#        k+=1
#        print(k)
#np.savetxt('./stiff_MM_2048.txt', MM_stiff)  

#print('Fundamental Mode')
#print('Soft')
#
#n=50
#masses = np.linspace(1.0,2.2,n)
#m1s, m2s = np.meshgrid(masses,masses)
#
#MM = np.zeros((n,n))
#
#k=0
#for j, m1 in enumerate(masses):
#    for i, m2 in enumerate(masses):
#        if m2>m1:
#            MM[i,j]= None
#            continue
#        L1 = soft_ml_interp(m1).item()
#        L2 = soft_ml_interp(m2).item()
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 2)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_true = hptilde.data.data - 1j*hctilde.data.data
#        
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_UR = hptilde.data.data - 1j*hctilde.data.data
#        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
#        MM[i,j] = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
#        k+=1
#        print(k)
#np.savetxt('./soft_MM_dyntides_2048.txt', MM) 
#
#print('Stiff')
#MM = np.zeros((n,n))
#k=0
#for j, m1 in enumerate(masses):
#    for i, m2 in enumerate(masses):
#        if m2>m1:
#            MM[i,j]= None
#            continue
#        L1 = stiff_ml_interp(m1).item()
#        L2 = stiff_ml_interp(m2).item()
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 3)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_true = hptilde.data.data - 1j*hctilde.data.data
#        
#        laldict = lal.CreateDict()
#        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
#        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
#        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
#        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
#        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
#        h_UR = hptilde.data.data - 1j*hctilde.data.data
#        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
#        MM_temp = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
#        MM[i,j] = MM_temp
#        k+=1
#        print(k)
#np.savetxt('./stiff_MM_dyntides_2048.txt', MM)           

print('Quadrupole Moment')

n=50
masses = np.linspace(1.0,2.2,n)
m1s, m2s = np.meshgrid(masses,masses)

k=0

soft_MM = np.zeros((n,n))
for j, m1 in enumerate(masses):
    for i, m2 in enumerate(masses):
        if m2>m1:
            soft_MM[i,j]= None
            continue
        L1 = soft_ml_interp(m1).item()
        L2 = soft_ml_interp(m2).item()
        laldict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 1)
        Q1 = get_quadrupole('soft', m1, 0.15)
        Q2 = get_quadrupole('soft', m2, 0.15)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(laldict, Q1)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon2(laldict, Q2)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2) 
        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0.15, 0., 0., 0.15, 1, incl,
        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
        h_true = hptilde.data.data - 1j*hctilde.data.data
        
        laldict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0.15, 0., 0., 0.15, 1, incl,
        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
        h_UR = hptilde.data.data - 1j*hctilde.data.data
        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
        soft_MM[i,j] = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
        k+=1
        print(k)
np.savetxt('./soft_MM_quad_2048.txt', soft_MM)      

print('Stiff')

k=0

stiff_MM = np.zeros((n,n))
for j, m1 in enumerate(masses):
    for i, m2 in enumerate(masses):
        if m2>m1:
            stiff_MM[i,j]= None
            continue
        L1 = stiff_ml_interp(m1).item()
        L2 = stiff_ml_interp(m2).item()
        laldict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 1)
        Q1 = get_quadrupole('stiff', m1, 0.15)
        Q2 = get_quadrupole('stiff', m2, 0.15)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(laldict, Q1)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon2(laldict, Q2)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2) 
        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0.15, 0., 0., 0.15, 1, incl,
        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
        h_true = hptilde.data.data - 1j*hctilde.data.data
        
        laldict = lal.CreateDict()
        lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)  
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)  
        hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0.15, 0., 0., 0.15, 1, incl,
        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
        h_UR = hptilde.data.data - 1j*hctilde.data.data
        freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
        stiff_MM[i,j] = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method = "Nelder-Mead")[0]
        k+=1
        print(k)
np.savetxt('./stiff_MM_quad_2048.txt', stiff_MM)  
