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

resonator = chip.Resonator(
    name="R",
    desc="Resonator",
    hilbert_dim=resonator_levels,
    params=parameters_resonator
)

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
    hamiltonian_func=hamiltonians.int_XX
)

drive_qubit_1 = chip.Drive(
    name="dQ1",
    desc="Qubit Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)

drive_qubit_2 = chip.Drive(
    name="dQ2",
    desc="Qubit Drive 2",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)


drive_resonator_1 = chip.Drive(
    name="dR1",
    desc="Resonator Drive 1",
    comment="Drive line on resonator",
    connected=["R"],
    hamiltonian_func=hamiltonians.x_drive
)

drive_resonator_2 = chip.Drive(
    name="dR2",
    desc="Resonator Drive 2",
    comment="Drive line on resonator",
    connected=["R"],
    hamiltonian_func=hamiltonians.x_drive
)

model = Mdl(
    [qubit, resonator], # Individual, self-contained components
    [drive_qubit_1, drive_qubit_2, drive_resonator_1, drive_resonator_2, qr_coupling]  # Interactions between components
)
model.set_lindbladian(False)
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
            "dQ1":{
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"]
            },
            "dR1":{
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"]
            },
            "dQ2":{
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"]
            },
            "dR2":{
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"]
            }
        }

    )

generator.devices["AWG"].enable_drag_2()
#%%

from qutip import *
import numpy as np
Nq = qubit_levels
Nr = resonator_levels
Ideal_gate = np.array(tensor(qeye(Nq), qeye(Nr)) + tensor(basis(Nq,1),basis(Nr,0))*tensor(basis(Nq,0),basis(Nr,1)).dag() + tensor(basis(Nq,0),basis(Nr,1))*tensor(basis(Nq,1),basis(Nr,0)).dag() - tensor(basis(Nq,1),basis(Nr,0))*tensor(basis(Nq,1),basis(Nr,0)).dag() - tensor(basis(Nq,0),basis(Nr,1))*tensor(basis(Nq,0),basis(Nr,1)).dag()) 
#print(Ideal_gate)
np.savetxt("ideal_gate.csv", Ideal_gate, delimiter=",")


t_swap_gate = 209e-9
sideband = 50e6

tswap_10_20 = 11e-9
tswap_20_01 = 190e-9


Qswap_params = {
    "amp": Qty(value=0.3,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=2.0e-9, min_val=0.0, max_val=tswap_10_20, unit="s"),
    "t_down": Qty(value=tswap_10_20-2.0e-9, min_val=0.0, max_val=tswap_10_20, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=tswap_10_20/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_swap_gate,min_val=0.1*t_swap_gate,max_val=1.5*t_swap_gate,unit="s")
}


Qswap2_params = {
    "amp": Qty(value=1.0,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=t_swap_gate - tswap_20_01, min_val=0.0, max_val=tswap_20_01, unit="s"),
    "t_down": Qty(value=t_swap_gate-2.0e-9, min_val=0.0, max_val=t_swap_gate, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=t_swap_gate/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_swap_gate,min_val=0.1*t_swap_gate,max_val=1.5*t_swap_gate,unit="s")
}

Qswap_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="Flattop pluse for SWAP gate",
    params=Qswap_params,
    shape=envelopes.flattop
)

Qswap2_pulse = pulse.Envelope(
    name="swap2_pulse",
    desc="Flattop pluse for SWAP gate",
    params=Qswap2_params,
    shape=envelopes.flattop
)

Rswap_params = {
    "amp": Qty(value=0.3,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=2.0e-9, min_val=0.0, max_val=tswap_10_20, unit="s"),
    "t_down": Qty(value=tswap_10_20-2.0e-9, min_val=0.0, max_val=tswap_10_20, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=tswap_10_20/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_swap_gate,min_val=0.1*t_swap_gate,max_val=1.5*t_swap_gate,unit="s")
}


Rswap2_params = {
    "amp": Qty(value=1.0,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=t_swap_gate - tswap_20_01, min_val=0.0, max_val=tswap_20_01, unit="s"),
    "t_down": Qty(value=t_swap_gate-2.0e-9, min_val=0.0, max_val=t_swap_gate, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=t_swap_gate/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_swap_gate,min_val=0.1*t_swap_gate,max_val=1.5*t_swap_gate,unit="s")
}


Rswap_pulse = pulse.Envelope(
    name="swap_pulse",
    desc="Flattop pluse for SWAP gate",
    params=Rswap_params,
    shape=envelopes.flattop
)

Rswap2_pulse = pulse.Envelope(
    name="swap2_pulse",
    desc="Flattop pluse for SWAP gate",
    params=Rswap2_params,
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


tlist = np.linspace(0,t_swap_gate, 1000)

drive_freq_qubit = 7650554480.090796
drive_freq_resonator = 7650554480.090796
carrier_freq = [drive_freq_qubit, drive_freq_resonator]
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


drive_freq_qubit = 9.5125e9
drive_freq_resonator = 9.5125e9
carrier_freq = [drive_freq_qubit, drive_freq_resonator]
carrier_parameters = {
            "Q":{"freq": Qty(value=carrier_freq[0], min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")},
            "R": {"freq": Qty(value=carrier_freq[1], min_val=0.0, max_val=10e9, unit="Hz 2pi"),
            "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad")}
            }

carriers_2 = [
    pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters["Q"]),
    pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters["R"])
]

ideal_gate = np.loadtxt("ideal_gate.csv", delimiter=",", dtype=np.complex128)

swap_gate = gates.Instruction(
    name="swap", targets=[0, 1], t_start=0.0, t_end=t_swap_gate, channels=["dQ1", "dR1", "dQ2", "dR2"],
    ideal = ideal_gate
)

swap_gate.set_ideal(ideal_gate)

swap_gate.add_component(Qswap_pulse, "dQ1")
swap_gate.add_component(carriers[0], "dQ1")

swap_gate.add_component(Qswap2_pulse, "dQ2")
swap_gate.add_component(carriers_2[0], "dQ2")

swap_gate.add_component(Rswap_pulse, "dR1")
swap_gate.add_component(carriers[1], "dR1")

swap_gate.add_component(Rswap2_pulse, "dR2")
swap_gate.add_component(carriers_2[1], "dR2")


gates_arr = [swap_gate]

#%%

Delta_1 = qubit_frequency - resonator_frequency
Delta_2 = (2 + qubit_anharm)*qubit_frequency
chi_0 = (coupling_strength**2)/Delta_1
chi_1 = (coupling_strength**2)/(Delta_2 - Delta_1)

carriers = createCarriers([resonator_frequency+sideband - chi_1/2, resonator_frequency+sideband - chi_1/2], sideband)

t_readout = 50e-9
t_total = 50e-9


readout_params = {
    "amp": Qty(value=2*np.pi*0.01,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=2e-9, min_val=0.0, max_val=t_readout, unit="s"),
    "t_down": Qty(value=t_readout-2.0e-9, min_val=0.0, max_val=t_readout, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=t_readout/2, unit="s"),
    "xy_angle": Qty(value=np.pi,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit=""),
    "t_final": Qty(value=t_total,min_val=0.1*t_total,max_val=1.5*t_total,unit="s")
}

readout_pulse = pulse.Envelope(
    name="readout-pulse",
    desc="Flattop pluse for SWAP gate",
    params=readout_params,
    shape=envelopes.flattop
)


tlist = np.linspace(0,t_total, 1000)
plotSignal(tlist, readout_pulse.shape(tlist, readout_pulse.params).numpy())

nodrive_pulse = pulse.Envelope(
    name="no_drive", 
    params={
        "t_final": Qty(
            value=t_total,
            min_val=0.5 * t_total,
            max_val=1.5 * t_total,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)

qubit_pulse = copy.deepcopy(readout_pulse)
qubit_pulse.params["amp"] = Qty(value=2*np.pi*0,min_val=0.0,max_val=10.0,unit="V")
qubit_pulse.params["xy_angle"] = Qty(value=0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad")
resonator_pulse = copy.deepcopy(readout_pulse)
resonator_pulse.params["amp"] = Qty(value=2*np.pi*0.01,min_val=0.0,max_val=10.0,unit="V")
resonator_pulse.params["xy_angle"] = Qty(value=-np.pi,min_val=-np.pi,max_val=2.5 * np.pi,unit="rad")

Readout_gate = gates.Instruction(
    name="Readout", targets=[1], t_start=0.0, t_end=t_total, channels=["dQ1", "dR1"]
)
Readout_gate.add_component(qubit_pulse, "dQ1")
Readout_gate.add_component(copy.deepcopy(carriers[0]), "dQ1")
Readout_gate.add_component(resonator_pulse, "dR1")
Readout_gate.add_component(copy.deepcopy(carriers[1]), "dR1")

gates_arr.append(Readout_gate)

#%%

parameter_map = PMap(instructions=gates_arr, model=model, generator=generator)
exp = Exp(pmap=parameter_map)
exp.set_opt_gates(["swap[0, 1]", 'Readout[1]'])
#%%

model.set_FR(False)
model.set_lindbladian(True)
exp.propagate_batch_size = 100

#%%
unitaries = exp.compute_propagators()
print(unitaries)

#%%

exp.write_config("Full_simulation.hjson")
parameter_map.store_values("Full_simulation_pmap_before_opt.c3log")

#%%

print("Plotting dynamics before optimization ... ")

psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,0)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ["swap[0, 1]", 'Readout[1]']
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Full_simulation_before_optimization.png")

t_sequence = t_swap_gate + t_readout
plotIQ(exp, sequence, model.ann_opers[1], resonator_frequency, resonator_frequency, t_sequence, spacing=100, usePlotly=False)
#%%

print("Starting optimization .... ")

parameter_map.set_opt_map([
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
    [("Readout[1]", "dR1", "carrier", "freq")],
    [("Readout[1]", "dR1", "readout-pulse", "amp")],
    [("Readout[1]", "dR1", "readout-pulse", "t_up")],
    [("Readout[1]", "dR1", "readout-pulse", "t_down")],
    [("Readout[1]", "dR1", "readout-pulse", "risefall")],
    [("Readout[1]", "dR1", "readout-pulse", "xy_angle")],
    [("Readout[1]", "dR1", "readout-pulse", "freq_offset")],
    [("Readout[1]", "dR1", "readout-pulse", "delta")],
    [("Readout[1]", "dQ1", "carrier", "freq")],
    [("Readout[1]", "dQ1", "readout-pulse", "amp")],
    [("Readout[1]", "dQ1", "readout-pulse", "t_up")],
    [("Readout[1]", "dQ1", "readout-pulse", "t_down")],
    [("Readout[1]", "dQ1", "readout-pulse", "risefall")],
    [("Readout[1]", "dQ1", "readout-pulse", "xy_angle")],
    [("Readout[1]", "dQ1", "readout-pulse", "freq_offset")],
    [("Readout[1]", "dQ1", "readout-pulse", "delta")],
])

parameter_map.print_parameters()

#%%
psi = [[0] * model.tot_dim]
ground_state_index = model.get_state_indeces([(0,0)])[0]
psi[0][ground_state_index] = 1
ground_state = tf.transpose(tf.constant(psi, tf.complex128))
if model.lindbladian:
    ground_state = tf_utils.tf_state_to_dm(ground_state)


psi = [[0] * model.tot_dim]
excited_state_index = model.get_state_indeces([(1,0)])[0]
psi[0][excited_state_index] = 1
excited_state = tf.transpose(tf.constant(psi, tf.complex128))
if model.lindbladian:
    excited_state = tf_utils.tf_state_to_dm(excited_state)


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

swap_cost = 1.0

psi_0 = excited_state

fid_params = {
    "ground_state": ground_state,
    "excited_state": excited_state,
    "a_rotated": a_rotated,
    "cutoff_distance": d_max,
    "psi_0": psi_0,
    "swap_cost": swap_cost,
    "lindbladian": model.lindbladian
}

#%%
opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.swap_and_readout,
    fid_subspace=["Q", "R"],
    pmap=parameter_map,
    algorithm=algorithms.lbfgs,
    options={"maxfun":250},
    run_name="swap_and_readout",
    fid_func_kwargs={"params":fid_params}
)
exp.set_opt_gates(["swap[0, 1]", "Readout[1]"])
opt.set_exp(exp)


#%%

tf.config.run_functions_eagerly(True)
opt.optimize_controls()
print(opt.current_best_goal)
print(parameter_map.print_parameters())

parameter_map.store_values("Full_simulation_pmap_after_opt.c3log")


plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Full_simulation_after_optimization.png")

t_sequence = t_swap_gate + t_readout
plotIQ(exp, sequence, model.ann_opers[1], resonator_frequency, resonator_frequency, t_sequence, spacing=100, usePlotly=False)

