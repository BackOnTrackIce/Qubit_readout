#%%
import os
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

from matplotlib import cm
from distinctipy import distinctipy

# %%

qubit_levels = 4
qubit_frequency = 5e9
qubit_anharm = -200e6
qubit_t1 = 20e-9
qubit_t2star = 40e-9
qubit_temp = 10e-6

qubit = chip.Qubit(
    name="Q",
    desc="Qubit",
    freq=Qty(value=qubit_frequency,min_val=1e9 ,max_val=8e9 ,unit='Hz 2pi'),
    anhar=Qty(value=qubit_anharm,min_val=-380e6 ,max_val=-120e6 ,unit='Hz 2pi'),
    hilbert_dim=qubit_levels,
    t1=Qty(value=qubit_t1,min_val=1e-9,max_val=90e-3,unit='s'),
    t2star=Qty(value=qubit_t2star,min_val=10e-9,max_val=90e-3,unit='s'),
    temp=Qty(value=qubit_temp,min_val=0.0,max_val=0.12,unit='K')
)

drive_qubit = chip.Drive(
    name="dQ",
    desc="Qubit Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)

model = Mdl(
    [qubit], # Individual, self-contained components
    [drive_qubit]  # Interactions between components
)
model.set_lindbladian(True)
model.set_dressed(False)

#%%

sim_res = 100e9
awg_res = 2e9
v2hz = 1e9

generator = Gnr(
        devices={
            "LO": devices.LO(name='lo', resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name='awg', resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac",
                resolution=sim_res,
                inputs=1,
                outputs=1
            ),
            "Response": devices.Response(
                name='resp',
                rise_time=Qty(
                    value=0.3e-9,
                    min_val=0.05e-9,
                    max_val=0.6e-9,
                    unit='s'
                ),
                resolution=sim_res,
                inputs=1,
                outputs=1
            ),
            "Mixer": devices.Mixer(name='mixer', inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name='v_to_hz',
                V_to_Hz=Qty(
                    value=1e9,
                    min_val=0.9e9,
                    max_val=1.1e9,
                    unit='Hz/V'
                ),
                inputs=1,
                outputs=1
            )
        },
        chains= {
            "dQ":{
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"]
            }
        }

    )


t_pulse = 10e-9
sideband = 50e6

nodrive_pulse = pulse.Envelope(
    name="no_drive",
    params={
        "t_final": Qty(
            value=t_pulse,
            min_val=0.5 * t_pulse,
            max_val=1.5 * t_pulse,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)

carrier_freq = qubit_frequency
carrier_parameters = {
            "freq": Qty(value=carrier_freq, min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")
            }

carrier = pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters)

No_drive_gate = gates.Instruction(
    name="nodrive", targets=[0], t_start=0.0, t_end=t_pulse, channels=["dQ"]
)
No_drive_gate.add_component(nodrive_pulse, "dQ")
No_drive_gate.add_component(carrier, "dQ")

#%%

t_pulse = 10e-9
sideband = 50e6


X_params = {
    "amp": Qty(value=1.0,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=2.0e-9, min_val=0.0, max_val=t_pulse, unit="s"),
    "t_down": Qty(value=t_pulse-2.0e-9, min_val=0.0, max_val=t_pulse, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=t_pulse/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_pulse,min_val=0.1*t_pulse,max_val=1.5*t_pulse,unit="s")
}


X_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="Flattop pluse for SWAP gate",
    params=X_params,
    shape=envelopes.flattop
)


carrier_freq = qubit_frequency
carrier_parameters = {
            "freq": Qty(value=carrier_freq, min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")
            }

carrier = pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters)

X_gate = gates.Instruction(
    name="x", targets=[0], t_start=0.0, t_end=t_pulse, channels=["dQ"]
)
X_gate.add_component(X_pulse, "dQ")
X_gate.add_component(carrier, "dQ")


#%%

parameter_map = PMap(instructions=[No_drive_gate, X_gate], model=model, generator=generator)
exp = Exp(pmap=parameter_map, sim_res=sim_res)

model.set_FR(False)
model.set_lindbladian(True)
exp.set_opt_gates(['nodrive[0]', 'x[0]'])

#%%
#compute_propagators_tf = tf.function(exp.compute_propagators)
#unitaries = compute_propagators_tf()
exp.set_prop_method("pwc")
unitaries = exp.compute_propagators()
#print(unitaries)

#%%

psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(0,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ['x[0]']
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False)

#%%

model.set_lindbladian(True)

psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ["nodrive[0]"]#['x[0]']


plotPopulationFromState(
                    exp, 
                    init_state, 
                    sequence, 
                    Num_shots=1, 
                    plot_avg=True, 
                    enable_vec_map=False,
                    batch_size=None,
                    states_to_plot=None,
                    solver="rk38"
)


# %%
model.set_lindbladian(False)
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ["nodrive[0]"]#['x[0]']

plotPopulationFromState(
                    exp, 
                    init_state, 
                    sequence, 
                    Num_shots=100, 
                    plot_avg=True, 
                    enable_vec_map=True,
                    batch_size=None,
                    states_to_plot=None,
                    solver="rk38"
                    
)
# %%
model.set_lindbladian(False)
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
#init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ['x[0]']

Num_shots = 1
result = exp.solve_stochastic_ode(
    init_state=init_state,
    sequence=sequence,
    Num_shots=Num_shots,
    enable_vec_map=False,
    batch_size=None,
    solver="rk38"
)
psis = result["states"]
ts = result["ts"]

#%%
#pops = []
#for rho in psis[0]:
#    pops.append(tf.math.real(tf.linalg.diag_part(rho)))
#
#plt.figure(figsize=[10,5])
#colors = distinctipy.get_colors(len(model.state_labels))
#plt.gca().set_prop_cycle(color=colors)
#plt.plot(ts, pops)
#plt.legend(
#    model.state_labels,
#    ncol=int(np.ceil(model.tot_dim / 15)),
#    bbox_to_anchor=(1.05, 1.0),
#    loc="upper left")
#plt.tight_layout()
#

#%%
#norms = tf.abs(tf.linalg.trace(psis))
#plt.figure(dpi=100)
#plt.plot(ts, tf.transpose(norms))
#plt.show()

# %%

#norms = tf.linalg.norm(psis, axis=2)
norms = tf.matmul(tf.transpose(psis, conjugate=True, perm=[0,1,3,2]), psis)
norms = tf.reshape(norms, [norms.shape[0], norms.shape[1]])
plt.figure(dpi=100)
#plt.plot(ts[0], tf.transpose(norms))
plt.plot(ts, tf.reduce_mean(norms, axis=0))
plt.show()

#%%

model.set_lindbladian(True)
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ['nodrive[0]']
result = exp.solve_lindblad_ode(
    init_state=init_state,
    sequence=sequence,
)
psis = result["states"]
ts = result["ts"]
#%%
norms = tf.linalg.trace(psis)
plt.figure(dpi=100)
#plt.plot(ts[0], tf.transpose(norms))
plt.plot(ts, norms)
plt.show()


# %%
# Testing interpolation
import c3.libraries.propagation as propagation
from scipy import interpolate

signal = generator.generate_signals(X_gate)
signals = []
for key in signal:
    signals.append(signal[key]["values"])
    ts = signal[key]["ts"]

sig_inter_fun = interpolate.interp1d(ts, signals[0], fill_value="extrapolate")

# %%

