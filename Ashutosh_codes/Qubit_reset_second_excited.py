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

import scipy as sp

# %%
qubit_levels = 3
qubit_frequency = 7.86e9
qubit_anharm = -264e6
qubit_t1 = 27e-6
qubit_t2star = 39e-6
qubit_temp = 50e-3

qubit = chip.Qubit(
    name="Q",
    desc="Qubit",
    freq=Qty(value=qubit_frequency,min_val=1e9 ,max_val=8e9 ,unit='Hz 2pi'),
    anhar=Qty(value=qubit_anharm,min_val=-380e6 ,max_val=-120e6 ,unit='Hz 2pi'),
    hilbert_dim=qubit_levels,
    t1=Qty(value=qubit_t1,min_val=1e-6,max_val=90e-6,unit='s'),
    t2star=Qty(value=qubit_t2star,min_val=10e-6,max_val=90e-3,unit='s'),
    temp=Qty(value=qubit_temp,min_val=0.0,max_val=0.12,unit='K')
)

# %%
resonator_levels = 3
resonator_frequency = 6.02e9
resonator_t1 = 27e-6
resonator_t2star = 39e-6
resonator_temp = 50e-3

parameters_resonator = {
    "freq": Qty(value=resonator_frequency,min_val=0e9 ,max_val=8e9 ,unit='Hz 2pi'),
    "t1": Qty(value=resonator_t1,min_val=1e-6,max_val=90e-6,unit='s'),
    "t2star": Qty(value=resonator_t2star,min_val=10e-6,max_val=90e-3,unit='s'),
    "temp": Qty(value=resonator_temp,min_val=0.0,max_val=0.12,unit='K')
}

resonator = chip.ReadoutResonator(
    name="R",
    desc="Resonator",
    hilbert_dim=resonator_levels,
    params=parameters_resonator
)

# %% [markdown]
# Coupling

# %%
coupling_strength = 130e6
qr_coupling = chip.Coupling(
    name="Q-R",
    desc="coupling",
    comment="Coupling qubit and resonator",
    connected=["Q", "R"],
    strength=Qty(
        value=coupling_strength,
        min_val=-1 * 1e3 ,
        max_val=200e6 ,
        unit='Hz 2pi'
    ),
    hamiltonian_func=hamiltonians.jaynes_cummings
)

# %% [markdown]
# Drives

# %%
drive_qubit = chip.Drive(
    name="dQ",
    desc="Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)
# %%
drive_resonator = chip.Drive(
    name="dR",
    desc="Drive 2",
    comment="Drive line on resonator",
    connected=["R"],
    hamiltonian_func=hamiltonians.x_drive
)

model = Mdl(
    [qubit, resonator], # Individual, self-contained components
    [drive_qubit, drive_resonator, qr_coupling]  # Interactions between components
)
model.set_lindbladian(False)
model.set_dressed(False)

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
            "dQ": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"],
            "dR": ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
        }
    )

generator.devices["AWG"].enable_drag_2()


# %% [markdown]
# Define SWAP gate for qubit and resonator by using Rabi oscillations

# %%
t_swap_gate = 200e-9
sideband = 50e6


swap_params = {
    "amp": Qty(value=1.0,min_val=0.1,max_val=10.0,unit="V"),
    "t_up": Qty(value=2.0e-9, min_val=0.1e-9, max_val=5.0e-9, unit="s"),
    "t_down": Qty(value=t_swap_gate-2.0e-9, min_val=t_swap_gate-5.0e-9, max_val=t_swap_gate-0.1e-9, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=5.0e-9, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_swap_gate,min_val=0.5*t_swap_gate,max_val=1.5*t_swap_gate,unit="s")
}

swap_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="Flattop pluse for SWAP gate",
    params=swap_params,
    shape=envelopes.flattop
)

nodrive_pulse = pulse.Envelope(
    name="no_drive",
    params={
        "t_final": Qty(
            value=t_swap_gate,
            min_val=0.5 * t_swap_gate,
            max_val=1.5 * t_swap_gate,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)

index = model.get_state_indeces([(2,0),(0,1)])
state_energies = [model.eigenframe[i].numpy() for i in index]
print(abs(state_energies[0] - state_energies[1])/(2*np.pi*1e9))

drive_freq = 9.5095e9
carrier_freq = [drive_freq, drive_freq]
carrier_parameters = {
            "Q":{"freq": Qty(value=carrier_freq[0], min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")},
            "R": {"freq": Qty(value=carrier_freq[1], min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")}
            }

carriers = [
    pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters["Q"]),
    pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters["R"])
]

qubit_pulse = copy.deepcopy(swap_pulse)
resonator_pulse = copy.deepcopy(swap_pulse)

ideal_gate = np.loadtxt("ideal_gate.csv", delimiter=",", dtype=np.complex128)
ideal_gate = tf.cast(ideal_gate, dtype=tf.complex128)

swap_gate = gates.Instruction(
    name="swap", targets=[0, 1], t_start=0.0, t_end=t_swap_gate, channels=["dQ", "dR"], 
    ideal= ideal_gate
)
swap_gate.add_component(qubit_pulse, "dQ")
swap_gate.add_component(copy.deepcopy(carriers[0]), "dQ")
swap_gate.add_component(resonator_pulse, "dR")
swap_gate.add_component(copy.deepcopy(carriers[1]), "dR")

gates_arr = [swap_gate]

# %%
init_state_index = model.get_state_indeces([(2,0)])[0]

# %%
print("----------------------------------------------")
print("------Simulating with unoptimized pulses------")
parameter_map = PMap(instructions=gates_arr, model=model, generator=generator)
exp = Exp(pmap=parameter_map)
exp.set_opt_gates(['swap[0, 1]'])
unitaries = exp.compute_propagators()
#plotComplexMatrix(unitaries['swap[0, 1]'].numpy())

# %%
psi_init = [[0] * model.tot_dim]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['swap[0, 1]']
states_to_plot = [(0,1), (1,0), (0,2), (2,0), (1,1)]
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, states_to_plot=states_to_plot, usePlotly=False, filename="Second_excited_Before_optimisation.png")

# %%

print("----------------------------------------------")
print("-----------Starting optimal control-----------")

parameter_map.set_opt_map([
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
])

parameter_map.print_parameters()

opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.state_transfer_infid_set_full,
    fid_subspace=["Q", "R"],
    pmap=parameter_map,
    algorithm=algorithms.lbfgs,
    options={"maxfun":200},
    run_name="SWAP_20_01",
    fid_func_kwargs={"psi_0":init_state}
)
exp.set_opt_gates(["swap[0, 1]"])
opt.set_exp(exp)

opt.optimize_controls()
print(opt.current_best_goal)
print(parameter_map.print_parameters())

# %%
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Second_excited_After_optimization.png")
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, states_to_plot=states_to_plot, filename="Second_excited_After_optimization_selected.png")

# %%
print("----------------------------------------------")
print("-----------Finished optimal control-----------")
