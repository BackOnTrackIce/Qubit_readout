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

tswap = 50e-9

exp = Exp()
exp.read_config("qubit_reset_experiment_short.hjson")
pmap = exp.pmap
model = pmap.model
pmap.load_values("current_vals_short.c3log")
ts = np.linspace(0, tswap, 100)

pmap_dict = pmap.asdict()
Q_pulse = pmap_dict["swap[0, 1]"]["drive_channels"]["dQ"]["swap_pulse"]
R_pulse = pmap_dict["swap[0, 1]"]["drive_channels"]["dR"]["swap_pulse"]

Qpulse_carrier = pmap_dict["swap[0, 1]"]["drive_channels"]["dQ"]["carrier"]
Rpulse_carrier = pmap_dict["swap[0, 1]"]["drive_channels"]["dR"]["carrier"]

Qpulse_shape = Q_pulse.shape(ts, Q_pulse.params)
Rpulse_shape = R_pulse.shape(ts, R_pulse.params)

Qpulse_pwc_params = {
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=tswap, min_val=10e-9, max_val=200e-9, unit="s"),
    "inphase": Qty(value=Qpulse_shape, min_val=0.0, max_val=1.0, unit=""),
    "t_final": Qty(value=tswap, min_val=10e-9, max_val=300e-9, unit="s")
}


Qpwc_swap_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="PWC pulse for Qubit",
    params=Qpulse_pwc_params,
    shape=envelopes.pwc_shape
)

Rpulse_pwc_params = {
    "t_bin_start": Qty(value=0.0, min_val=0.0, max_val=1e-9, unit="s"),
    "t_bin_end": Qty(value=tswap, min_val=10e-9, max_val=200e-9, unit="s"),
    "inphase": Qty(value=Rpulse_shape, min_val=0.0, max_val=1.0, unit=""),
    "t_final": Qty(value=tswap, min_val=10e-9, max_val=300e-9, unit="s") 
}


Rpwc_swap_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="PWC pulse for resonator",
    params=Rpulse_pwc_params,
    shape=envelopes.pwc_shape
)

ideal_gate = pmap_dict["swap[0, 1]"]["ideal"]

#%%

swap_gate = gates.Instruction(
    name="swap", targets=[0, 1], t_start=0.0, t_end=tswap+1e-9, channels=["dQ", "dR"], 
    ideal=ideal_gate
)
swap_gate.add_component(Qpwc_swap_pulse, "dQ")
swap_gate.add_component(Qpulse_carrier, "dQ")
swap_gate.add_component(Rpwc_swap_pulse, "dR")
swap_gate.add_component(Rpulse_carrier, "dR")

gates_arr = [swap_gate]

generator = pmap.generator
pmap = PMap(instructions=gates_arr, model=model, generator=generator)
exp = Exp(pmap=pmap)
exp.write_config("qubit_reset_experiment_short_pwc.hjson")


#%%
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
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, states_to_plot=states_to_plot, usePlotly=False, filename="Short_swap_Before_opt.png" )

#%%
print("----------------------------------------------")
print("-----------Starting optimal control-----------")

"""
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
"""

opt_map = [
    [("swap[0, 1]", "dR", "carrier", "freq")],
    [("swap[0, 1]", "dR", "swap_pulse", "inphase")],
    [("swap[0, 1]", "dQ", "carrier", "freq")],
    [("swap[0, 1]", "dQ", "swap_pulse", "inphase")]
]

pmap.set_opt_map(opt_map)
#%%
opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.state_transfer_infid_set_full,
    fid_subspace=["Q", "R"],
    pmap=pmap,
    algorithm=algorithms.lbfgs,
    options={"maxfun":1000},
    run_name="SWAP_20_01_short",
    fid_func_kwargs={"psi_0":init_state}
)
exp.set_opt_gates(["swap[0, 1]"])
opt.set_exp(exp)

opt.optimize_controls()
print(opt.current_best_goal)
print(pmap.print_parameters())


plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Short_swap_After_opt.png")
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, states_to_plot=states_to_plot, filename="Short_swap_After_opt_selected.png")

pmap.store_values("current_vals_short_pwc.c3log")

print("----------------------------------------------")
print("-----------Finished optimal control-----------")


# %%
