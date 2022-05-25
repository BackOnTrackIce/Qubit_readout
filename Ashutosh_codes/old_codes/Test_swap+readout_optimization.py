# %%
import os
from re import I
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
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
import c3.libraries.tasks as tasks

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.fidelities as fidelities
import c3.libraries.envelopes as envelopes
import c3.utils.qt_utils as qt_utils
import c3.utils.tf_utils as tf_utils
from c3.optimizers.optimalcontrol import OptimalControl

import plotly.graph_objects as go
from plotting import *
from utilities_functions import *

import scipy as sp

# %%

exp = Exp()
exp.read_config("qubit_reset_experiment_two_carriers.hjson")
pmap = exp.pmap
model = pmap.model
pmap.load_values("best_point.txt")

exp.set_opt_gates(['swap[0, 1]'])
unitaries = exp.compute_propagators()

init_state_index = model.get_state_indeces([(1,0)])[0]
psi_init = [[0] * model.tot_dim]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['swap[0, 1]']
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Swap.png")


resonator_frequency = 6.02e9
t_total = 250e-9

# %%


opt_map = [
    [("swap[0, 1]", "dR1", "carrier", "freq")],
    [("swap[0, 1]", "dR1", "swap_pulse", "amp")],
    [("swap[0, 1]", "dR1", "swap_pulse", "t_up")],
    [("swap[0, 1]", "dR1", "swap_pulse", "t_down")],
    [("swap[0, 1]", "dR1", "swap_pulse", "risefall")],
    [("swap[0, 1]", "dR1", "swap_pulse", "xy_angle")],
    [("swap[0, 1]", "dR1", "swap_pulse", "freq_offset")],
    [("swap[0, 1]", "dR1", "swap_pulse", "delta")],
    [("swap[0, 1]", "dQ1", "carrier", "freq")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "amp")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "t_up")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "t_down")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "risefall")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "xy_angle")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "freq_offset")],
    [("swap[0, 1]", "dQ1", "swap_pulse", "delta")],
    [("swap[0, 1]", "dR2", "carrier", "freq")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "amp")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "t_up")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "t_down")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "risefall")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "xy_angle")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "freq_offset")],
    [("swap[0, 1]", "dR2", "swap2_pulse", "delta")],
    [("swap[0, 1]", "dQ2", "carrier", "freq")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "amp")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "t_up")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "t_down")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "risefall")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "xy_angle")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "freq_offset")],
    [("swap[0, 1]", "dQ2", "swap2_pulse", "delta")],
]


pmap.set_opt_map(opt_map)

pmap.print_parameters()

psi = [[0] * model.tot_dim]
ground_state_index = model.get_state_indeces([(0,0)])[0]
psi[0][ground_state_index] = 1
ground_state = tf.transpose(tf.constant(psi, tf.complex128))

psi = [[0] * model.tot_dim]
excited_state_index = model.get_state_indeces([(1,0)])[0]
psi[0][excited_state_index] = 1
excited_state = tf.transpose(tf.constant(psi, tf.complex128))

freq_drive = resonator_frequency

aR = tf.convert_to_tensor(model.ann_opers[1], dtype = tf.complex128)
aQ = tf.convert_to_tensor(model.ann_opers[0], dtype = tf.complex128)
aR_dag = tf.transpose(aR, conjugate=True)
NR = tf.matmul(aR_dag,aR)
aQ_dag = tf.transpose(aQ, conjugate=True)
NQ = tf.matmul(aQ_dag, aQ)

Urot = tf.linalg.expm(1j*2*np.pi*freq_drive*(NR + NQ)*t_total)
U_rot_dag = tf.transpose(Urot, conjugate=True)
a_rotated = tf.matmul(U_rot_dag, tf.matmul(aR, Urot))

d_max = 1.0

psi_0 = excited_state

swap_cost = 10.0


params = {
    "ground_state": ground_state,
    "excited_state": excited_state,
    "a_rotated": a_rotated,
    "cutoff_distance": d_max, 
    "psi_0": psi_0,
    "swap_cost": swap_cost
}


opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.swap_and_readout,
    fid_subspace=["Q", "R"],
    pmap=pmap,
    algorithm=algorithms.lbfgs,
    options={"maxfun":1000},
    run_name="Readout_IQ",
    fid_func_kwargs={"params":params}
)
exp.set_opt_gates(["swap[0, 1]"])
opt.set_exp(exp)

exp.write_config("Test_swap+readout_run1.hjson")

opt.optimize_controls()
print(opt.current_best_goal)
print(pmap.print_parameters())

pmap.store_values("Test_swap+readout_optimized_values_run1.c3log")
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Swap_after_opt.png")
