import lalsimulation
import lal
import bilby
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from multiprocessing import Pool
from scipy.optimize import minimize

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

    count = 5
    counter = 0
    cguess  = np.array([0.,0.])    
    #off = [0,0.0001,0.002,0.005,0.005,0.1,0.5,1.0,2.,3.,10]

    global_minmismatch = 1.0
    best_method = None

    while global_minmismatch > 1e-2 and counter <= count:
        for imethod in method:
            res = minimize(_mismatch, cguess, method=imethod)
            mismatch_val = _mismatch(res.x)
            if mismatch_val < global_minmismatch:
                global_minmismatch = mismatch_val
                best_method = imethod
        cguess = cguess + 0.02
        counter = counter + 1.  
        #print(minmismatch)
    return global_minmismatch, best_method

def dot(f, h1, h2):
    df = f[1]-f[0]
    result = np.real(4.*df*np.nansum(np.conjugate(h1)*h2/1))
    return result

def compute_mismatch_pair_BL(args):
    m1, m2, EOS = args
    if m2 > m1:
        return np.nan, (m1,m2)
    if EOS == 'soft':
        EOS_func = soft_ml_interp
    elif EOS == 'stiff':
        EOS_func = stiff_ml_interp
    else:
        raise ValueError("EOS must be 'soft' or 'stiff'")
    # True waveform
    L1 = EOS_func(m1).item()
    L2 = EOS_func(m2).item()
    h_true = generate_waveform(m1, m2, L1, L2, incl, deltaF, fmin, fmax, fref, approx)

    # Universal-relation waveform
    Ls = 0.5*(L1+L2)
    La_UR = bilby.gw.conversion.binary_love_fit_lambda_symmetric_mass_ratio_to_lambda_antisymmetric(Ls, m2/m1)
    L1_UR, L2_UR = bilby.gw.conversion.lambda_symmetric_lambda_antisymmetric_to_lambda_1_lambda_2(Ls, La_UR)
    h_UR = generate_waveform(m1, m2, L1_UR, L2_UR, incl, deltaF, fmin, fmax, fref, approx)

    freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
    mismatch = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method="Nelder-Mead")[0]

    return mismatch, (m1, m2)

def generate_waveform(m1, m2, L1, L2, incl, deltaF, fmin, fmax, fref, approx):
    laldict = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, 0)
    lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, 0)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)
    
    hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., 0., 0., 0., 0., 1, incl,
        0., 0., 0., 0., deltaF, fmin, fmax, fref, laldict, approx)
    
    return hptilde.data.data - 1j*hctilde.data.data

approx = lalsimulation.IMRPhenomXAS_NRTidalv3
incl = np.pi
srate = 4096
deltaF = 1/srate
fmin = 5.
fref = fmin
fmax = srate/2
outdir = '/work/williams5/testing-urs/bilby/scripts/mismatches'

eos_data = np.loadtxt('/work/williams5/testing-urs/EOS/soft_mrlfmode.dat')
eos_mass = eos_data[:,1]
eos_lambda = eos_data[:,2]

soft_ml_interp = ius(eos_mass, eos_lambda)

eos_data = np.loadtxt('/work/williams5/testing-urs/EOS/stiff_mrlfmode.dat')
eos_mass = eos_data[:,1]
eos_lambda = eos_data[:,2]

stiff_ml_interp = ius(eos_mass, eos_lambda)

n=200
n_cores = 32
masses = np.linspace(1.0, 2.2, n)

EOS = 'soft'

# Create list of argument tuples
mass_pairs_grid = [(m1, m2, EOS) for j, m1 in enumerate(masses)
                                 for i, m2 in enumerate(masses)]

print('Computing Soft Mismatches')

with Pool(n_cores) as pool:
    results = pool.map(compute_mismatch_pair_BL, mass_pairs_grid)

MM = np.full((n, n), np.nan)
mass_pairs_out = []

for mismatch, (m1, m2) in results:
    i = np.where(masses == m2)[0][0]
    j = np.where(masses == m1)[0][0]
    MM[i, j] = mismatch
    mass_pairs_out.append((m1, m2))


np.savetxt(f'{outdir}/{EOS}_MM_BL.txt', MM)
np.savetxt(f'{outdir}/{EOS}_MM_BL_masses.txt', np.array(mass_pairs_out))

print('Computing Stiff Mismatches')

EOS = 'stiff'

# Create list of argument tuples
mass_pairs_grid = [(m1, m2, EOS) for j, m1 in enumerate(masses)
                                 for i, m2 in enumerate(masses)]

with Pool(n_cores) as pool:
    results = pool.map(compute_mismatch_pair_BL, mass_pairs_grid)

MM = np.full((n, n), np.nan)
mass_pairs_out = []

for mismatch, (m1, m2) in results:
    i = np.where(masses == m2)[0][0]
    j = np.where(masses == m1)[0][0]
    MM[i, j] = mismatch
    mass_pairs_out.append((m1, m2))


np.savetxt(f'{outdir}/{EOS}_MM_BL.txt', MM)
np.savetxt(f'{outdir}/{EOS}_MM_BL_masses.txt', np.array(mass_pairs_out))
