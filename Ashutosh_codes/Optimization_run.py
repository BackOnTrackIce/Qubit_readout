#%%
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from c3.c3objs import Quantity as Qty
from c3.parametermap import ParameterMap as PMap
from c3.experiment import Experiment as Exp
from c3.model import Model as Mdl
from c3.generator.generator import Generator as Gnr

# Building blocks
import c3.generator.devices as devices
import c3.signal.gates as gates
import c3.libraries.chip as chip
import c3.signal.pulse as pulse

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes
from c3.optimizers.optimalcontrol import OptimalControl

from plotting import *
from utilities_functions import *
#%%

exp = Exp()
exp.read_config("qubit_reset_experiment.hjson")
pmap = exp.pmap
model = pmap.model


print("----------------------------------------------")
print("------Simulating with unoptimized pulses------")


exp.set_opt_gates(['swap[0, 1]'])
unitaries = exp.compute_propagators()

init_state_index = model.get_state_indeces([(1,0)])[0]
psi_init = [[0] * model.tot_dim]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['swap[0, 1]']

states_to_plot = [(0,1), (1,0), (0,2), (2,0), (1,1)]
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, states_to_plot=states_to_plot, usePlotly=False, filename="Full_swap_Before_optimisation.png")


print("----------------------------------------------")
print("-----------Starting optimal control-----------")

opt_map = [
    [("swap[0, 1]", "dR", "carrier", "freq")],
    [("swap[0, 1]", "dR", "swap_pulse", "amp")],
    [("swap[0, 1]", "dR", "swap_pulse", "t_up")],
    [("swap[0, 1]", "dR", "swap_pulse", "t_down")],
    [("swap[0, 1]", "dR", "swap_pulse", "risefall")],
    [("swap[0, 1]", "dR", "swap_pulse", "xy_angle")],
    [("swap[0, 1]", "dR", "swap_pulse", "freq_offset")],
    [("swap[0, 1]", "dR", "swap_pulse", "delta")],
    [("swap[0, 1]", "dQ", "carrier", "freq")],
    [("swap[0, 1]", "dQ", "swap_pulse", "amp")],
    [("swap[0, 1]", "dQ", "swap_pulse", "t_up")],
    [("swap[0, 1]", "dQ", "swap_pulse", "t_down")],
    [("swap[0, 1]", "dQ", "swap_pulse", "risefall")],
    [("swap[0, 1]", "dQ", "swap_pulse", "xy_angle")],
    [("swap[0, 1]", "dQ", "swap_pulse", "freq_offset")],
    [("swap[0, 1]", "dQ", "swap_pulse", "delta")],
]

pmap.set_opt_map(opt_map)

opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.state_transfer_infid_set_full,
    fid_subspace=["Q", "R"],
    pmap=pmap,
    algorithm=algorithms.lbfgs,
    options={"maxfun":200},
    run_name="SWAP_20_01_full",
    fid_func_kwargs={"psi_0":init_state}
)
exp.set_opt_gates(["swap[0, 1]"])
opt.set_exp(exp)

opt.optimize_controls()
print(opt.current_best_goal)
print(pmap.print_parameters())


plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Full_swap_After_optimization.png")
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, states_to_plot=states_to_plot, filename="Full_swap_After_optimization_selected.png")

pmap.store_values("current_vals.c3log")

print("----------------------------------------------")
print("-----------Finished optimal control-----------")


# %%
