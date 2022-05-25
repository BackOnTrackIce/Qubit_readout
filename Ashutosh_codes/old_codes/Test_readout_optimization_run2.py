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
exp.read_config("Test_readout_optimization.hjson")
pmap = exp.pmap
model = pmap.model
pmap.load_values("test_IQ_opt_values.txt")
exp = Exp(pmap=pmap)
exp.set_opt_gates(['Readout[1]'])
model.set_lindbladian(False)
unitaries = exp.compute_propagators()


resonator_frequency = 6.02e9
t_total = 50e-9

# %%
pmap.set_opt_map([
    [("Readout[1]", "dR", "carrier", "freq")],
    [("Readout[1]", "dR", "readout-pulse", "amp")],
    [("Readout[1]", "dR", "readout-pulse", "t_up")],
    [("Readout[1]", "dR", "readout-pulse", "t_down")],
    [("Readout[1]", "dR", "readout-pulse", "risefall")],
    [("Readout[1]", "dR", "readout-pulse", "xy_angle")],
    [("Readout[1]", "dR", "readout-pulse", "freq_offset")],
    [("Readout[1]", "dR", "readout-pulse", "delta")],
    [("Readout[1]", "dQ", "carrier", "freq")],
    [("Readout[1]", "dQ", "readout-pulse", "amp")],
    [("Readout[1]", "dQ", "readout-pulse", "t_up")],
    [("Readout[1]", "dQ", "readout-pulse", "t_down")],
    [("Readout[1]", "dQ", "readout-pulse", "risefall")],
    [("Readout[1]", "dQ", "readout-pulse", "xy_angle")],
    [("Readout[1]", "dQ", "readout-pulse", "freq_offset")],
    [("Readout[1]", "dQ", "readout-pulse", "delta")],
])

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

d_max = 3.0

fid_params = {
    "ground_state": ground_state,
    "excited_state": excited_state,
    "a_rotated": a_rotated,
    "cutoff_distance": d_max
}

opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.IQ_plane_distance,
    fid_subspace=["Q", "R"],
    pmap=pmap,
    algorithm=algorithms.lbfgs,
    options={"maxfun":250},
    run_name="Readout_IQ",
    fid_func_kwargs={"params":fid_params}
)
exp.set_opt_gates(["Readout[1]"])
opt.set_exp(exp)

exp.write_config("Test_readout_optimization_run2.hjson")

opt.optimize_controls()
print(opt.current_best_goal)
print(pmap.print_parameters())

pmap.store_values("Test_readout_optimization_values_run2.c3log")
