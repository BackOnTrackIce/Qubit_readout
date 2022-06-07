#%%
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
#%%
exp = Exp()
exp.read_config("Qubit_below_resonator.hjson")
parameter_map = exp.pmap
model = parameter_map.model

parameter_map.load_values("Qubit_below_resonator_after_opt.c3log")
unitaries = exp.compute_propagators()
init_state_index = model.get_state_indeces([(0,0)])[0]
psi_init = [[0] * model.tot_dim]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ["swap_10_20[0, 1]", "swap_20_01[0, 1]"]

#%%
print("Starting optimization .... ")

parameter_map.set_opt_map([
    [("swap_10_20[0, 1]", "dR", "carrier", "freq")],
    [("swap_10_20[0, 1]", "dQ", "carrier", "freq")],
    [("swap_20_01[0, 1]", "dR", "carrier", "freq")],
    [("swap_20_01[0, 1]", "dQ", "carrier", "freq")],
])

parameter_map.print_parameters()

#%%
psi = [[0] * model.tot_dim]
excited_state_index = model.get_state_indeces([(1,0)])[0]
psi[0][excited_state_index] = 1
excited_state = tf.transpose(tf.constant(psi, tf.complex128))
if model.lindbladian:
    excited_state = tf_utils.tf_state_to_dm(excited_state)

psi_0 = excited_state

fid_params = {
    "psi0": psi_0,
    "lindbladian": model.lindbladian
}

#%%
opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.state_transfer_infid_set_full,
    fid_subspace=["Q", "R"],
    pmap=parameter_map,
    algorithm=algorithms.cma_pre_lbfgs,
    run_name="swap_and_readout",
    fid_func_kwargs={"params":fid_params}
)
exp.set_opt_gates(["swap_10_20[0, 1]", "swap_20_01[0, 1]"])
opt.set_exp(exp)

#%%
opt.optimize_controls()
print(opt.current_best_goal)
print(parameter_map.print_parameters())

parameter_map.store_values("Qubit_below_resonator_after_opt.c3log")

# %%
