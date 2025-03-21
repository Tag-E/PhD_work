import numpy as np
import h5py as h5
import gvar as gv
from pathlib import Path
import correlatoranalyser as ca 

#from mpl_style import *
from matplotlib.style import *
import matplotlib.pyplot as plt
COLORS = {0:"red",
          1:"blue",
          2:"green",
          3:"purple",
          4:"brown",
          5:"black",
          6:"pink",}


## original initialization

# file = Path(".../....h5")

# src_snk_seps = [11,14,16,19]
# Nbst = 100

# Ratios_bst = {}

# with h5.File(file,'r') as h5f: 
#     for t in src_snk_seps:
#         # Nbst, 0-momentum, tau
#         Ratios_bst[t] = h5f[f"/Ratios/polarisation_None/current_direction_t/avg/src_snk_sep{t}/resample"][:,0,:].real * 0.74740



## my code to load the data
import os
from moments_toolkit import moments_toolkit, ratio_formula
from utilities import bootstrap_resamples, jackknife_resamples

p3fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW_extended/3PointCorrelation/"
p2fold = os.environ['mount_point_path'] + "48c48/binned_1012_hmz370_BMW_extended/2PointCorrelation/"

opAnalyzer = moments_toolkit(p3fold, p2fold,
                            skip3p=False, skipop=False,
                            verbose=True,
                            fast_data_folder = "fast_data_extended_p0_q0",
                            operator_folder= "operator_database",
                            momentum='PX0_PY0_PZ0',
                            insertion_momentum = 'qx0_qy0_qz0',
                            tag_2p='hspectrum',
                            max_n=2 #max_n=3
                            )

opV1 = 1/6 * opAnalyzer.get_operator(2)
opAnalyzer.append_operator(opV1)
#opA2 = 1/np.sqrt(2) * opAnalyzer.get_operator(32)
#opAnalyzer.append_operator(opA2)


opAnalyzer.remove_T(3,4,5,12)

src_snk_seps = opAnalyzer.chosen_T_list

#We first take the 3 point and 2 point correlators needed to compute the ratio
p3_corr = opAnalyzer.get_p3corr() #shape = (nop, nconf, nT, maxT+1)
p2_corr = opAnalyzer.get_p2corr() #shape = (nconf, latticeT)

#the shape of the ratio is given by (nT, maxT+1), i.e.
#we instantiate the output ratio
#Rmean = np.zeros(shape=R_shape, dtype=float) 

Nbst = 100
Ratios_bst = {}

#we loop over all the T values we have
for iT,T in enumerate(opAnalyzer.chosen_T_list):

    #we perform the jackknife analysis (the observable being the ratio we want to compute)
    Ratios_bst[T] = bootstrap_resamples([p3_corr[0,:,iT,:], p2_corr], lambda x,y: ratio_formula(x,y, T=T, gauge_axis=0), res_axis_list=[0,0], Nres=Nbst)[:,:T+1]
    #Ratios_bst[T] = jackknife_resamples([p3_corr[0,:,iT,:], p2_corr], lambda x,y: ratio_formula(x,y, T=T, gauge_axis=0), res_axis_list=[0,0])[:,:T+1]

## rest of the code

class SymmetricRatioModel:
    def __init__(self, number_states_sink, number_states_source, include_mix_term = True):
        self.number_states_sink = number_states_sink
        self.number_states_source = number_states_source
        self.include_mix_term = include_mix_term
        
        # GS
        if   number_states_sink == 1 and number_states_source == 1:
            self.Nparams = 1
        # GS + ES @ sink 
        elif number_states_sink == 2 and number_states_source == 1:
            self.Nparams = 3
        # GS + ES @ source
        elif number_states_sink == 1 and number_states_source == 2:
            self.Nparams = 3
        # GS + ES @ source + ES @ sink + ES @ (source & sink)
        elif number_states_sink == 2 and number_states_source == 2 and include_mix_term:
            self.Nparams = 4
        # GS + ES @ source + ES @ sink + ES @ (source & sink)
        elif number_states_sink == 2 and number_states_source == 2:
            self.Nparams = 3
        else:
            raise NotImplementedError("Model only implemented for at most one excited state at source and/or sink")

    def __call__(self, t_tau, p):
        r"""
            parameter:
                - E-m = E_N(q)-m_N > 0 (Ground state exponent)
                - dE{n}(q) = E_n(q) - E_N(q) > 0 (relative energy/mass if q=0)
                    - Currently accepted: dE1(q), dE1(0)
                - Amn (replace m,n by integers, matrix element for the mth excited state at source, and nth excited state a sink)
                    - A00 is the ground state matrix element
        """
        t = t_tau[:,0]
        tau = t_tau[:,1]

        # The ground state contribution (exponential is factorized)
        out = np.full_like(tau, p['A00'], dtype = object)

        # The excited state at source 
        if self.number_states_source == 2:
            out += p["A01"] * np.exp(                        - tau * p["dE1(0)"])

        # The excited state at sink
        if self.number_states_sink == 2:
            out += p["A01"] * np.exp( -(t-tau) * p["dE1(0)"]                    ) 

        # The excited states at source and sink 
        if self.number_states_source == 2 and self.number_states_sink == 2 and self.include_mix_term:
            out += p["A11"] * np.exp( -(t-tau) * p["dE1(0)"] - tau * p["dE1(0)"])  

        return out

    def prior(self, **kwargs):
        prior = gv.BufferDict()

        prior["A00"] = gv.gvar(1,0.5) 
        
        # The excited state at source 
        if self.number_states_source == 2:
            prior["log(dE1(0))"] = gv.log(gv.gvar(0.1, 1))
            prior["A01"]    = gv.gvar(1e-2,1)

        # The excited state at sink
        if self.number_states_sink == 2:
            prior["log(dE1(0))"] = gv.log(gv.gvar(0.1, 1))
            prior["A01"]    = gv.gvar(1e-2,1)

        # The excited states at source and sink 
        if self.number_states_source == 2 and self.number_states_sink == 2 and self.include_mix_term:
            prior["A11"]    = gv.gvar(0.1,1)   
        
        return prior

model = SymmetricRatioModel(number_states_sink=2,number_states_source=2, include_mix_term = True)

prior = model.prior()

tau_s, tau_e = 2, -2

abscissa = np.array([
    [t, tau] for t in src_snk_seps for tau in np.arange(1,t)[tau_s:tau_e]
])

Ratio_ror = np.zeros( (Nbst, len(abscissa)) )

for idx, (t,tau) in enumerate(abscissa):
    Ratio_ror[:,idx] = Ratios_bst[t][:,tau]

fit_result = ca.fit(
    abscissa = abscissa,
    ordinate_est = np.mean( Ratio_ror, axis = 0 ),
    ordinate_std = np.std ( Ratio_ror, axis = 0 ),
    ordinate_cov = np.cov ( Ratio_ror, rowvar = False ),
    resample_ordinate_est = Ratio_ror,
    resample_ordinate_std = np.std ( Ratio_ror, axis = 0 ),
    resample_ordinate_cov = np.cov ( Ratio_ror, rowvar = False ),
    
    central_value_fit = True,
    central_value_fit_correlated = True,
    resample_fit = True,
    resample_fit_correlated = True,
    resample_fit_resample_prior = False,
    resample_type = "bst",
    # Fitting infos
    model = model,
    prior = prior,
)

print(fit_result)

for t_id, t in enumerate(src_snk_seps):
    taus = np.arange(1,t)

    plt.errorbar(
        taus - t/2,
        np.mean( Ratios_bst[t][:,1:-1], axis = 0 ),
        np.std ( Ratios_bst[t][:,1:-1], axis = 0 ),
        fmt = '.:',
        capsize = 2,
        label = f"${t=}$",
        color = list(COLORS.values())[t_id]
    )

    abscissa = np.array([
        (t, tau) for tau in taus[tau_s:tau_e]
    ])

    fit_ordinate = fit_result.eval( abscissa )

    plt.plot(taus[tau_s:tau_e]-t/2, fit_ordinate["est"], color = list(COLORS.values())[t_id])
    
    plt.fill_between(
        taus[tau_s:tau_e]-t/2, 
        fit_ordinate["est"]+fit_ordinate["err"],
        fit_ordinate["est"]-fit_ordinate["err"],
        color = list(COLORS.values())[t_id],
        alpha = 0.2
    )

plt.xlabel(r"$\tau - t/2$")
plt.ylabel(r"$R(t,\tau)$")
plt.show()
        
