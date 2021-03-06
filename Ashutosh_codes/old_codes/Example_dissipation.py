#%%
import os
from re import I
import numpy as np
import copy
from typing import List

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


#%%
qubit_levels = 3
qubit_frequency = 5e9
qubit_anharm = -200e6
qubit_t1 = 20e-9
qubit_t2star = 40e-9
qubit_temp = 50e-3

qubit = chip.Qubit(
    name="Q",
    desc="Qubit",
    freq=Qty(value=qubit_frequency,min_val=1e9 ,max_val=8e9 ,unit='Hz 2pi'),
    anhar=Qty(value=qubit_anharm,min_val=-380e6 ,max_val=-120e6 ,unit='Hz 2pi'),
    hilbert_dim=qubit_levels,
    t1=Qty(value=qubit_t1,min_val=10e-9,max_val=90e-9,unit='s'),
    t2star=Qty(value=qubit_t2star,min_val=10e-9,max_val=90e-9,unit='s'),
    temp=Qty(value=qubit_temp,min_val=0.0,max_val=0.12,unit='K')
)

drive_qubit = chip.Drive(
    name="dQ",
    desc="Drive Q",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)

model = Mdl(
    [qubit], # Individual, self-contained components
    [drive_qubit]  # Interactions between components
)
model.set_dressed(False)
model.set_lindbladian(True)


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
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"]
            }
        }
    )


sideband = 50e6
t_pulse = 100e-9
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

parameter_map = PMap(instructions=[No_drive_gate], model=model, generator=generator)
exp = Exp(pmap=parameter_map)

model.set_FR(False)
model.set_lindbladian(True)
exp.set_opt_gates(['nodrive[0]'])
exp.propagate_batch_size = None
#%%
unitaries = exp.compute_propagators()
print(unitaries)
#%%

def calculatePopulation(
    exp: Exp, psi_init: tf.Tensor, sequence: List[str]
) -> np.array:
    """
    Calculates the time dependent population starting from a specific initial state.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: tf.Tensor
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state

    Returns
    -------
    np.array
       two-dimensional array, first dimension: time, second dimension: population of the levels
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    dUs = exp.partial_propagators
    if model.lindbladian:
        psi_t = tf_utils.tf_dm_to_vec(psi_init).numpy()
    else:
        psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in sequence:
        for du in dUs[gate]:
            psi_t = np.matmul(du, psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)
    return pop_t


def plotPopulation(
    exp: Exp,
    psi_init: tf.Tensor,
    sequence: List[str],
    labels: List[str] = None,
    filename: str = None,
):
    """
    Plots time dependent populations. They need to be calculated with `runTimeEvolution` first.
    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: np.array
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state
    labels: List[str]
        Optional list of names for the levels. If none, the default list from the experiment will be used.
    states_to_plot: List[tuple]
        List of str for the states to plot. If none, all the states would be plotted. 
    usePlotly: bool
        Whether to use Plotly or Matplotlib
    vertical_lines: bool
        If true, this add a dotted vertical line after each gate
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    pop_t = calculatePopulation(exp, psi_init, sequence)

    # timestamps
    dt = exp.ts[1] - exp.ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])

    legend_labels = labels if labels else model.state_labels
    labelX = "Time [ns]"
    labelY = "Population"

    # create the plot
    fig, axs = plt.subplots(1, 1, figsize=[10, 5])

    axs.plot(ts / 1e-9, pop_t.T)

    # set plot properties
    axs.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
    axs.set_xlabel(labelX)
    axs.set_ylabel(labelY)
    plt.legend(
        legend_labels,
        ncol=int(np.ceil(model.tot_dim / 15)),
        bbox_to_anchor=(1.05, 1.0),
        loc="upper left",
    )
    plt.tight_layout()


    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()

#%%
psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(1,)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ['nodrive[0]']
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence)

# %%
