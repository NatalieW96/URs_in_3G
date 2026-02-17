import lalsimulation
import lal
import bilby
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from multiprocessing import Pool
from scipy.optimize import minimize
from tqdm import tqdm
import os
import pickle


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

def compute_mismatch_pair_dyntides(args):
    m1, m2, EOS = args
    if m2 > m1:
        return np.nan, (m1,m2)
    if EOS == 'soft':
        EOS_func = soft_ml_interp
        f = 2
    elif EOS == 'stiff':
        EOS_func = stiff_ml_interp
        f = 3
    else:
        raise ValueError("EOS must be 'soft' or 'stiff'")
    # True waveform
    L1 = EOS_func(m1).item()
    L2 = EOS_func(m2).item()

    # True waveform with F-modes enabled
    h_true = generate_waveform_dyn(m1, m2, L1, L2, Fmodes=f)

    # UR waveform (no F-modes)
    h_UR = generate_waveform_dyn(m1, m2, L1, L2, Fmodes=0)

    freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
    mismatch = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method="Nelder-Mead")[0]

    return mismatch, (m1, m2)

def compute_mismatch_pair_quadrupole(args):
    m1, m2, EOS = args
    if m2 > m1:
        return np.nan, (m1, m2)
    
    if EOS == 'soft':
        EOS_func = soft_ml_interp
    elif EOS == 'stiff':
        EOS_func = stiff_ml_interp
    else:
        raise ValueError("EOS must be 'soft' or 'stiff'")

    L1 = EOS_func(m1).item()
    L2 = EOS_func(m2).item()

    # True waveform (UnivRel=1, spins, quadrupole)
    h_true = generate_waveform_quadrupole(m1, m2, L1, L2, EOS, Fmodes=0, univrel=1, chi1=0.15, chi2=0.15)

    # UR waveform (UnivRel=0, same spins/quadrupole ignored)
    h_UR = generate_waveform_quadrupole(m1, m2, L1, L2, EOS, Fmodes=0, univrel=0, chi1=0.15, chi2=0.15)

    freqs = np.arange(0, len(h_UR)*deltaF, deltaF)
    mismatch = get_mismatch(freqs, h_true, h_UR, fmin, fmax, method="Nelder-Mead")[0]

    return mismatch, (m1, m2)

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

def generate_waveform_quadrupole(m1, m2, L1, L2, EOS, Fmodes=0, univrel=0, chi1=0., chi2=0.):
    laldict = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertFModesFlag(laldict, Fmodes)
    lalsimulation.SimInspiralWaveformParamsInsertUnivRelFlag(laldict, univrel)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(laldict, L1)
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(laldict, L2)

    # Insert quadrupole moments only if UnivRel=1
    if univrel == 1:
        Q1 = get_quadrupole(EOS, m1, chi1)
        Q2 = get_quadrupole(EOS, m2, chi2)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon1(laldict, Q1)
        lalsimulation.SimInspiralWaveformParamsInsertdQuadMon2(laldict, Q2)

    hptilde, hctilde = lalsimulation.SimInspiralChooseFDWaveform(
        m1*lal.MSUN_SI, m2*lal.MSUN_SI, 0., 0., chi1, 0., 0., chi2, 1, incl,
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
n_cores = 128
masses = np.linspace(1.0, 2.2, n)


checkpoint_file_soft = f"{outdir}/soft_checkpoint.npy"
checkpoint_file_stiff = f"{outdir}/stiff_checkpoint.npy"

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                processed_results = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print(f"Warning: checkpoint file {checkpoint_file} is empty or corrupted. Starting fresh.")
            processed_results = []
        processed_pairs = {pair for _, pair in processed_results}
    else:
        processed_results = []
        processed_pairs = set()
    return processed_results, processed_pairs

def process_mass_pairs(mass_pairs_grid, compute_func, n_cores, checkpoint_file, desc="Computing Mismatches"):
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                processed_results = pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print(f"Warning: checkpoint {checkpoint_file} is empty or corrupted. Starting fresh.")
            processed_results = []
        processed_pairs = {pair for _, pair in processed_results}
    else:
        processed_results = []
        processed_pairs = set()

    # Prepare list of mass pairs not yet processed
    remaining_pairs = [p for p in mass_pairs_grid if (p[0], p[1]) not in processed_pairs]

    total_remaining = len(remaining_pairs)
    print(f"{desc}: {len(processed_results)} already done, {total_remaining} remaining")

    # Create a single tqdm progress bar for all remaining work
    with tqdm(total=total_remaining, desc=desc, unit="pairs") as pbar:
        batch_size = 5000  # adjust based on memory/runtime
        for batch_start in range(0, total_remaining, batch_size):
            batch = remaining_pairs[batch_start:batch_start+batch_size]
            if not batch:
                continue

            with Pool(n_cores) as pool:
                for result in pool.imap(compute_func, batch, chunksize=1):
                    processed_results.append(result)
                    processed_pairs.add(result[1])
                    pbar.update(1)  # update progress bar for each completed pair

            # Save checkpoint after each batch
            with open(checkpoint_file, "wb") as f:
                pickle.dump(processed_results, f)

    return processed_results

EOS = 'soft'

# Create list of argument tuples
mass_pairs_grid = [(m1, m2, EOS) for j, m1 in enumerate(masses)
                                 for i, m2 in enumerate(masses)]

print('Computing Soft Mismatches')


results = process_mass_pairs(
    mass_pairs_grid,
    compute_mismatch_pair_quadrupole,
    n_cores,
    checkpoint_file_soft,
    desc="Soft Mismatches"
)

MM = np.full((n, n), np.nan)
mass_pairs_out = []

for mismatch, (m1, m2) in results:
    i = np.where(masses == m2)[0][0]
    j = np.where(masses == m1)[0][0]
    MM[i, j] = mismatch
    mass_pairs_out.append((m1, m2))


np.savetxt(f'{outdir}/{EOS}_MM_quad.txt', MM)
np.savetxt(f'{outdir}/{EOS}_MM_quad_masses.txt', np.array(mass_pairs_out))

print('Computing Stiff Mismatches')

EOS = 'stiff'

# Create list of argument tuples
mass_pairs_grid = [(m1, m2, EOS) for j, m1 in enumerate(masses)
                                 for i, m2 in enumerate(masses)]

results = process_mass_pairs(
    mass_pairs_grid,
    compute_mismatch_pair_quadrupole,
    n_cores,
    checkpoint_file_stiff,
    desc="Stiff Mismatches"
)

MM = np.full((n, n), np.nan)
mass_pairs_out = []

for mismatch, (m1, m2) in results:
    i = np.where(masses == m2)[0][0]
    j = np.where(masses == m1)[0][0]
    MM[i, j] = mismatch
    mass_pairs_out.append((m1, m2))


np.savetxt(f'{outdir}/{EOS}_MM_quad.txt', MM)
np.savetxt(f'{outdir}/{EOS}_MM_quad_masses.txt', np.array(mass_pairs_out))
