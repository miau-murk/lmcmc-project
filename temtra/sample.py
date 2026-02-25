from .conf_calc.angles import Conformation
from .conf_calc.conf_calc import ConfCalc
from rdkit import Chem
import numpy as np
from temtra import energy
import temtra.littlemcmc.littlemcmc as lmc
from temtra import collector
from temtra.ttr import HMCParams, HMC, TemperedTransitions

################################################################################
# XTB FUNCTION
################################################################################

ref_conf = Chem.MolFromMolFile("butan.mol", removeHs=False)
all_dih_angles = Conformation.find_unique_dihedral_angles(ref_conf)
nonring_dih_angles = all_dih_angles[0]
rotatable_dih_idx = [list(dih_angle[0]) for dih_angle in nonring_dih_angles]
calculator = ConfCalc(
    mol=ref_conf,
    dir_to_xyzs="xtb_calcs/",
    rotable_dihedral_idxs=rotatable_dih_idx,
)

def logp_dlogp_xtb(phi, E0=-13.652247): # дописать!
    phi = np.asarray(phi, dtype=float)
    
    kB_au = 3.1668105e-6
    T = 300.0
    beta = 1.0 / (kB_au * T)

    result = calculator.get_energy(
        phi.tolist(),
        req_opt=False,
        req_grad=True,
    )
    E_phi = result["energy"]
    log_p = -beta * (E_phi - E0)

    grad_E_phi = np.array([g[1] for g in result["grads"]])
    grad_E_phi = np.asarray(grad_E_phi, dtype=float)
    grad_logp = -beta * grad_E_phi  

    return log_p, grad_logp


def build_grid(beta_min, n):
    q = (beta_min)**(1/(n-1))
    grid = [round(q**k, 4) for k in range(n)]
    return grid

if __name__ == "__main__":

    
    sampling_func = logp_dlogp_xtb
    temped_cycles = 1
    hmc_draws_nstep = 10
    hmc_tunes_nstep = 0
    hmc_train_nsteps = 10
    ndim = 1
    chains = 1
    starts = [np.array([0.0 for _ in range(ndim)], dtype=float) for __ in range(chains)]


    print("======== HMC SETTINGS ESTIMATION ========")

    init_potential = lmc.quadpotential.QuadPotentialDiagAdapt(ndim, 
                                                initial_mean=np.mean(np.array(starts), axis=0),
                                                initial_diag=None)

    hmc_settings = HMCParams(step_scale=0.25,
                             path_length=2.0,
                             max_steps=4,
                             target_accept=0.8,
                             adapt_step_size=True
                             )
    
    hmc_tune = HMC(
        logp_dlogp_func=sampling_func,
        ndim=ndim,
        chains=chains,
        params=hmc_settings,
        potential=init_potential
    )
    
    trace_train, stats_train = hmc_tune.draw(starts=starts, draws=1000, tune=500)
    pot = hmc_tune.step.potential
    mass_diag_estim = np.asarray(pot._var, dtype=float).copy()
    step_size_estim = hmc_tune.step.step_size

    print("======== STRARTING TEMPERED TRANSITION HMC ========")

    potential = lmc.quadpotential.QuadPotentialDiag(mass_diag_estim)
    

    tt_settings = HMCParams(step_scale=step_size_estim,
                            path_length=4.0,
                            max_steps=8,
                            target_accept=0.6,
                            adapt_step_size=False
                            )
    
    hmc_settings = HMCParams(step_scale=step_size_estim,
                             path_length=2.0,
                             max_steps=4,
                             target_accept=0.8,
                             adapt_step_size=False
                             )
    
    hmc = HMC(
        logp_dlogp_func=sampling_func,
        ndim=ndim,
        chains=chains,
        params=hmc_settings,
        potential=potential
    )


    temp_transition = TemperedTransitions(
        base_logp_dlogp=sampling_func,
        betas=build_grid(0.05, 15),
        ndim=ndim,
        chains=chains,
        params=tt_settings,
        potential=potential
    )

    TRACES = []
    STATS = []
    INFO = []

    for k in range(temped_cycles):

        trace, stat = hmc.draw(starts=starts, draws=hmc_draws_nstep, tune=hmc_tunes_nstep)
        tt_starts = trace[:, -1, :]

        if k != (temped_cycles+1): # последний шаг без tempered transition

            print(f"----- {k+1} CYCLE OF TEMPERING -----")
            x_next, info = temp_transition.step(tt_starts, return_list=False)

            print(x_next)
            print(info["accept"])

            starts = x_next.copy()

        TRACES.append(trace)
        STATS.append(stat)
        INFO.append(info)

    
    TRACE_ALL = collector.concat_traces(TRACES)
    STATS_ALL = collector.concat_stats(STATS, chains=chains, total=TRACE_ALL.shape[1])

    collector.save_to_npz(
        "tempered_hmc_run.npz",
        trace=TRACE_ALL,
        stats=STATS_ALL,
        draws=temped_cycles * hmc_draws_nstep,
        tune=temped_cycles * hmc_tunes_nstep,
        discard_tuned_samples=False,
        meta={"temped_cycles": temped_cycles, "betas": build_grid(0.01, 15)},
    )

    # collector.save_to_npz(
    #     "train.npz",
    #     trace=[trace_train],
    #     stats=[stats_train],
    #     draws=hmc_train_nsteps,
    #     tune=0,
    #     discard_tuned_samples=False,
    #     meta={"temped_cycles": temped_cycles, "betas": build_grid(0.05, 10)},
    # )
