#%%
import os
import tempfile
import numpy as np
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
# Main C3 objects
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

#%matplotlib widget

import plotly.graph_objects as go
from plotting import *
from utilities_functions import *

# %%
qubit_levels = 3
qubit_frequency = 4e9
qubit_anharm = -210e6
qubit_t1 = 27e-6
qubit_t2star = 39e-6
qubit_temp = 50e-3

qubit = chip.Qubit(
    name="Q",
    desc="Qubit",
    freq=Qty(value=qubit_frequency,min_val=2e9 ,max_val=8e9 ,unit='Hz 2pi'),
    anhar=Qty(value=qubit_anharm,min_val=-380e6 ,max_val=-120e6 ,unit='Hz 2pi'),
    hilbert_dim=qubit_levels,
    t1=Qty(value=qubit_t1,min_val=1e-6,max_val=90e-6,unit='s'),
    t2star=Qty(value=qubit_t2star,min_val=10e-6,max_val=90e-3,unit='s'),
    temp=Qty(value=qubit_temp,min_val=0.0,max_val=0.12,unit='K')
)

# %%
resonator_levels = 10
resonator_frequency = 5e9
resonator_t1 = 27e-6
resonator_t2star = 39e-6
resonator_temp = 50e-3

parameters_resonator = {
    "freq": Qty(value=resonator_frequency,min_val=2e9 ,max_val=8e9 ,unit='Hz 2pi'),
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

# %%

coupling_strength = 20e6
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
# %%

drive_qubit = chip.Drive(
    name="dQ",
    desc="Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)
drive_resonator = chip.Drive(
    name="dR",
    desc="Drive 2",
    comment="Drive line on resonator",
    connected=["R"],
    hamiltonian_func=hamiltonians.x_drive
)

drives = [drive_qubit, drive_resonator]
# %%
init_temp = 50e-3
init_ground = tasks.InitialiseGround(
    init_temp=Qty(
        value=init_temp,
        min_val=-0.001,
        max_val=0.22,
        unit='K'
    )
)
# %%

model = Mdl(
    [qubit, resonator], # Individual, self-contained components
    [drive_qubit, drive_resonator, qr_coupling]  # Interactions between components
)
model.set_lindbladian(False)
model.set_dressed(False)
# %%

sim_res = 100e9
awg_res = 2e9
v2hz = 1e9

generator = createGenerator(drives) 
generator.devices["AWG"].enable_drag_2()

# %%
plotComplexMatrix(qr_coupling.get_Hamiltonian())
plotComplexMatrix(resonator.get_Hamiltonian())
plotComplexMatrix(qubit.get_Hamiltonian())
plotComplexMatrix(model.get_Hamiltonian())
# %%

# Defining single qubit X gates on the qubit

t_final = 15e-9
sideband = 50e6
gauss_params = {
    "amp": Qty(value=0.5,min_val=0.2,max_val=0.6,unit="V"),
    "t_final": Qty(value=t_final,min_val=0.5 * t_final,max_val=1.5 * t_final,unit="s"),
    "sigma": Qty(value=t_final / 4,min_val=t_final / 8,max_val=t_final / 2,unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit="")
}

gauss_pulse = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp for single qubit gates",
    params=gauss_params,
    shape=envelopes.gaussian_nonorm
)

nodrive_pulse = pulse.Envelope(
    name="no_drive",
    params={
        "t_final": Qty(
            value=t_final,
            min_val=0.5 * t_final,
            max_val=1.5 * t_final,
            unit="s"
        )
    },
    shape=envelopes.no_drive
)

## Create carriers
qubit_freqs = model.get_qubit_freqs()
carriers = createCarriers(qubit_freqs, sideband)
# %%

qubit_pulse = copy.deepcopy(gauss_pulse)
resonator_pulse = copy.deepcopy(nodrive_pulse)
X_gate = gates.Instruction(
    name="x", targets=[0], t_start=0.0, t_end=t_final, channels=["dQ", "dR"]
)
X_gate.add_component(qubit_pulse, "dQ")
X_gate.add_component(copy.deepcopy(carriers[0]), "dQ")
X_gate.add_component(resonator_pulse, "dR")
X_gate.add_component(copy.deepcopy(carriers[1]), "dR")

single_q_gates = [X_gate]
# %%

parameter_map = PMap(instructions=single_q_gates, model=model, generator=generator)
exp = Exp(pmap=parameter_map)
exp.set_opt_gates(['x[0]'])
unitaries = exp.compute_propagators()
plotComplexMatrix(unitaries['x[0]'].numpy())
# %%

psi_init = [[0] * model.tot_dim]
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
sequence = ['x[0]']
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=True)
# %%

# Optimisation of the pulse

parameter_map.set_opt_map([
    [("x[0]", "dQ", "gauss", "amp")],
    [("x[0]", "dQ", "gauss", "freq_offset")],
    [("x[0]", "dQ", "gauss", "xy_angle")],
    [("x[0]", "dQ", "gauss", "delta")],
    [("x[0]", "dR", "carrier", "framechange")]
])

parameter_map.print_parameters()

opt = OptimalControl(
    dir_path="./output/",
    fid_func=fidelities.average_infid_set,
    fid_subspace=["Q", "R"],
    pmap=parameter_map,
    algorithm=algorithms.lbfgs,
    options={"maxfun": 150},
    run_name="better_x"
)
exp.set_opt_gates(["x[0]"])
opt.set_exp(exp)

# %%

# Run the optimisation
opt.optimize_controls()
opt.current_best_goal
# %%
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=True)
# %%

# Implement readout of resonator
# readout pulse on resonator

def calculateState(
    exp: Experiment,
    psi_init: tf.Tensor,
    sequence: List[str]     
):

    """
    Calculates the state of system with time.

    Parameters
    ----------
    exp: Experiment,
        The experiment containing the model and propagators
    psi_init: tf.Tensor,
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state

    Returns
    -------
    psi_list: List[tf.Tensor]
        List of states
    """

    model = exp.pmap.model
    dUs = exp.partial_propagators
    psi_t = psi_init.numpy()
    psi_list = [psi_t]
    for gate in sequence:
        for du in dUs[gate]:
            psi_list.append(np.matmul(du, psi_t))
    
    return np.array(psi_list)


psi_list = calculateState(exp, init_state, sequence)


#%%
def plotIQ(
        exp: Experiment,
        psi_init: tf.Tensor, 
        sequence: List[str], 
        annihilation_operator: tf.Tensor
):
    
    """
    Calculate and plot the I-Q values for resonator 

    Parameters
    ----------
    exp: Experiment,

    psi_init: tf.Tensor,
 
    sequence: List[str], 

    annihilation_operator: tf.Tensor


    Returns
    -------
        
    """

    dt = exp.ts[1] - exp.ts[0]
    psi_list = calculateState(exp, psi_init, sequence)
    ts = np.linspace(0.0, dt * psi_list.shape[0], psi_list.shape[0])
    
    rho_list = np.array([tf_utils.tf_state_to_dm(i) for i in psi_list])
    expect_val = np.array([tf_utils.tf_measure_operator(annihilation_operator, i) for i in rho_list])
    Q = np.real(expect_val)
    I = np.imag(expect_val)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = Q, y = I, mode = "lines"))
    fig.show()

plotIQ(exp, init_state, sequence, model.ann_opers[1])




# %%
