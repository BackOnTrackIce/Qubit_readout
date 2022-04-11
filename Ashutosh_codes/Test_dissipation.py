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

resonator_levels = 3 #10
resonator_frequency = 6.02e9
resonator_t1 = 235e-9
resonator_t2star = 39e-6
resonator_temp = 50e-3

parameters_resonator = {
    "freq": Qty(value=resonator_frequency,min_val=0e9 ,max_val=8e9 ,unit='Hz 2pi'),
    "t1": Qty(value=resonator_t1,min_val=100e-9,max_val=1e-6,unit='s'),
    "t2star": Qty(value=resonator_t2star,min_val=10e-6,max_val=90e-3,unit='s'),
    "temp": Qty(value=resonator_temp,min_val=0.0,max_val=0.12,unit='K')
}

resonator = chip.ReadoutResonator(
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
    hamiltonian_func=hamiltonians.jaynes_cummings
)

drive_qubit = chip.Drive(
    name="dQ",
    desc="Drive 1",
    comment="Drive line on qubit",
    connected=["Q"],
    hamiltonian_func=hamiltonians.x_drive
)

drives = [drive_qubit]

model = Mdl(
    [qubit, resonator], # Individual, self-contained components
    [drive_qubit, qr_coupling]  # Interactions between components
)
model.set_dressed(False)
model.set_lindbladian(False)

# %%

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
            "QuadraturesToValues": devices.QuadraturesToValues(name="quad_to_val", inputs=1, outputs=1),
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
            "dQ":["AWG", "DigitalToAnalog", "Response", "QuadraturesToValues", "VoltsToHertz"],
            "R": ["AWG", "DigitalToAnalog", "Response", "QuadraturesToValues", "VoltsToHertz"]
        }
    )

generator.devices["AWG"].enable_drag_2()

# %%
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
    if model.lindbladian:
        psi_t = tf_utils.tf_dm_to_vec(psi_init)
    else:
        psi_t = psi_init
    psi_list = [psi_t]
    for gate in sequence:
        for du in dUs[gate]:
            psi_t = tf.matmul(du, psi_t)
            psi_list.append(psi_t)

    return psi_list

def calculateExpectationValue(states, Op, lindbladian):
    expect_val = []
    for i in states:
        if lindbladian:
            expect_val.append(tf.linalg.trace(tf.matmul(Op, i)))
        else:
            expect_val.append(tf.matmul(tf.matmul(tf.transpose(i, conjugate=True), Op),i)[0,0])
    return expect_val

def plotNumberOperator(
    exp: Experiment, 
    init_state: tf.Tensor,
    sequence: List[str]
):
    model = exp.pmap.model
    psi_list = calculateState(exp, init_state, sequence)

    aR = tf.convert_to_tensor(model.ann_opers[1], dtype=tf.complex128)
    aR_dag = tf.transpose(aR, conjugate=True)
    NR = tf.matmul(aR_dag,aR)
    expect_val_R = calculateExpectationValue(psi_list, NR)

    aQ = tf.convert_to_tensor(model.ann_opers[0], dtype=tf.complex128)
    aQ_dag = tf.transpose(aQ, conjugate=True)
    NQ = tf.matmul(aQ_dag, aQ)
    expect_val_Q = calculateExpectationValue(psi_list, NQ)


    ts = exp.ts

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = ts, y = np.real(expect_val_R), mode = "lines", name="Resonator"))
    fig.add_trace(go.Scatter(x = ts, y = np.real(expect_val_Q), mode = "lines", name="Qubit"))
    fig.show()


def frameRotation(frame_op, freq, ts, psi_list, lindbladian):
    psi_rotated = []
    
    for i in range(len(psi_list)):
        U = tf.linalg.expm(1j*2*np.pi*freq*(frame_op)*ts[i])
        if lindbladian:
            U_dag = tf.linalg.expm(-1j*2*np.pi*freq*(frame_op)*ts[i])
            rho_i = tf_utils.tf_vec_to_dm(psi_list[i])
            psi_rotated.append(tf.matmul(tf.matmul(U_dag, rho_i), U))    ## Check this. Now it is U_dag*rho*U but I think it should be U*rho*U_dag
        else:
            psi_rotated.append(tf.matmul(U, psi_list[i]))

    return psi_rotated


def frameOfDrive(exp, psi_list, freq):
    model = exp.pmap.model
    aR = tf.convert_to_tensor(model.ann_opers[1], dtype = tf.complex128)
    aQ = tf.convert_to_tensor(model.ann_opers[0], dtype = tf.complex128)

    n = len(psi_list)

    aR_dag = tf.transpose(aR, conjugate=True)
    NR = tf.matmul(aR_dag,aR)

    aQ_dag = tf.transpose(aQ, conjugate=True)
    NQ = tf.matmul(aQ_dag, aQ)

    dt = (exp.ts[1] - exp.ts[0]).numpy()
    ts = tf.linspace(0.0, dt*n, n)
    
    I = tf.eye(len(aR), dtype=tf.complex128)
    
    """
    psi_rotated = []
    
    for i in range(n):
        U = tf.linalg.expm(1j*2*np.pi*freq*(NR + NQ)*ts[i])
        if model.lindbladian:
            U_dag = tf.linalg.expm(-1j*2*np.pi*freq*(NR + NQ)*ts[i])
            rho_i = tf_utils.tf_vec_to_dm(psi_list[i])
            psi_rotated.append(tf.matmul(tf.matmul(U_dag, rho_i), U))    ## Check this. Now it is U_dag*rho*U but I think it should be U*rho*U_dag
        else:
            psi_rotated.append(tf.matmul(U, psi_list[i]))

    return psi_rotated
    """

    return frameRotation(NR+NQ, freq, tf.cast(ts, dtype=tf.complex128), psi_list, model.lindbladian)

def plotIQ(
        exp: Experiment, 
        sequence: List[str], 
        annihilation_operator: tf.Tensor
):
    
    """
    Calculate and plot the I-Q values for resonator 

    Parameters
    ----------
    exp: Experiment,
 
    sequence: List[str], 

    annihilation_operator: tf.Tensor


    Returns
    -------
        
    """
    model = exp.pmap.model
    annihilation_operator = tf.convert_to_tensor(annihilation_operator, dtype=tf.complex128)
    
    state_index = exp.pmap.model.get_state_index((0,0))
    psi_init_0 = [[0] * model.tot_dim]
    psi_init_0[0][state_index] = 1
    init_state_0 = tf.transpose(tf.constant(psi_init_0, tf.complex128))
    
    if model.lindbladian:
        init_state_0 = tf_utils.tf_state_to_dm(init_state_0)

    psi_list = calculateState(exp, init_state_0, sequence)
    psi_list_0 =  frameOfDrive(exp, psi_list, resonator_frequency)
    expect_val_0 = calculateExpectationValue(psi_list_0, annihilation_operator, model.lindbladian)
    Q0 = np.real(expect_val_0)
    I0 = np.imag(expect_val_0)
    

    state_index = exp.pmap.model.get_state_index((1,0))
    psi_init_1 = [[0] * model.tot_dim]
    psi_init_1[0][state_index] = 1
    init_state_1 = tf.transpose(tf.constant(psi_init_1, tf.complex128))
    
    if model.lindbladian:
        init_state_1 = tf_utils.tf_state_to_dm(init_state_1)

    psi_list = calculateState(exp, init_state_1, sequence)
    psi_list_1 =  frameOfDrive(exp, psi_list, resonator_frequency)
    expect_val_1 = calculateExpectationValue(psi_list_1, annihilation_operator, model.lindbladian)
    Q1 = np.real(expect_val_1)
    I1 = np.imag(expect_val_1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = Q0, y = I0, mode = "lines", name="Ground state"))
    fig.add_trace(go.Scatter(x = Q1, y = I1, mode = "lines", name ="Excited state"))
    fig.write_image("Readout_IQ.png")


def plotIQ_testing(
        exp: Experiment, 
        sequence: List[str], 
        annihilation_operator: tf.Tensor
):
    
    """
    Calculate and plot the I-Q values for resonator 

    Parameters
    ----------
    exp: Experiment,
 
    sequence: List[str], 

    annihilation_operator: tf.Tensor


    Returns
    -------
        
    """
    model = exp.pmap.model
    annihilation_operator = tf.convert_to_tensor(annihilation_operator, dtype=tf.complex128)
    
    state_index = exp.pmap.model.get_state_index((0,0))
    psi_init_0 = [[0] * model.tot_dim]
    psi_init_0[0][state_index] = 1
    init_state_0 = tf.transpose(tf.constant(psi_init_0, tf.complex128))
    
    psi_list = calculateState(exp, init_state_0, sequence)
    psi_list_0 =  frameOfDrive(exp, psi_list, resonator_frequency)
    expect_val_0 = calculateExpectationValue(psi_list_0, annihilation_operator)
    Q0 = np.real(expect_val_0)
    I0 = np.imag(expect_val_0)
    

    state_index = exp.pmap.model.get_state_index((0,1))
    psi_init_1 = [[0] * model.tot_dim]
    psi_init_1[0][state_index] = 1
    init_state_1 = tf.transpose(tf.constant(psi_init_1, tf.complex128))
    
    psi_list = calculateState(exp, init_state_1, sequence)
    psi_list_1 =  frameOfDrive(exp, psi_list, resonator_frequency)
    expect_val_1 = calculateExpectationValue(psi_list_1, annihilation_operator)
    Q1 = np.real(expect_val_1)  
    I1 = np.imag(expect_val_1)

    #fig = go.Figure()
    #fig.add_trace(go.Scatter(x = Q0, y = I0, mode = "lines", name="Ground state"))
    #fig.add_trace(go.Scatter(x = Q1, y = I1, mode = "lines", name ="Excited state"))
    #fig.show()
    plt.plot(Q0, I0, label="Ground state")
    plt.plot(Q1, I1, label="Excited state")
    plt.legend()



# %%
qubit_freqs = model.get_qubit_freqs()
sideband = 50e6
carriers = createCarriers([0.0, 0.0], sideband)

t_readout = 1e-9
t_total = 10e-9
sideband = 50e6


readout_params = {
    "amp": Qty(value=2*np.pi*0.005,min_val=0.0,max_val=100.0,unit="V"),
    "t_final": Qty(value=t_readout,min_val=0.5 * t_readout,max_val=1.5 * t_readout,unit="s"),
    "sigma": Qty(value=t_readout/5,min_val=t_readout/20,max_val=t_readout/2,unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
    "freq_offset": Qty(value=-sideband - 3e6,min_val=-56 * 1e6,max_val=-52 * 1e6,unit="Hz 2pi"),
    "delta": Qty(value=-1,min_val=-5,max_val=3,unit="")
}

readout_pulse = pulse.Envelope(
    name="readout",
    desc="Gaussian pulse for readout",
    params=readout_params,
    shape=envelopes.gaussian_nonorm
)

"""
readout_params = {
    "amp": Qty(value=1.0,min_val=0.0,max_val=10.0,unit="V"),
    "t_up": Qty(value=2e-9, min_val=0.0, max_val=t_readout, unit="s"),
    "t_down": Qty(value=t_readout-2.0e-9, min_val=0.0, max_val=t_readout, unit="s"),
    "risefall": Qty(value=1.0e-9, min_val=0.1e-9, max_val=t_readout/2, unit="s"),
    "xy_angle": Qty(value=0.0,min_val=-0.5 * np.pi,max_val=2.5 * np.pi,unit="rad"),
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
"""
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

qubit_pulse = copy.deepcopy(nodrive_pulse)
resonator_pulse = copy.deepcopy(readout_pulse)
Readout_gate = gates.Instruction(
    name="Readout", targets=[1], t_start=0.0, t_end=t_total, channels=["R", "dQ"]
)
Readout_gate.add_component(qubit_pulse, "dQ")
Readout_gate.add_component(copy.deepcopy(carriers[0]), "dQ")
Readout_gate.add_component(resonator_pulse, "R")
Readout_gate.add_component(copy.deepcopy(carriers[1]), "R")

readout_gates = [Readout_gate]

# %%
parameter_map = PMap(instructions=readout_gates, model=model, generator=generator)
exp = Exp(pmap=parameter_map)

model.use_FR = False
exp.use_control_fields = False
model.set_lindbladian(True)
exp.set_opt_gates(['Readout[1]'])
print("----------Calculating Unitaries----------")
unitaries = exp.compute_propagators()

psi_init = [[0] * model.tot_dim]
init_state_index = model.get_state_indeces([(0,1)])[0]
psi_init[0][init_state_index] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
print("Lindbladian = ", model.lindbladian)
if model.lindbladian:
    init_state = tf_utils.tf_state_to_dm(init_state)
sequence = ['Readout[1]']
print("----------Plotting Population----------")
plotPopulation(exp=exp, psi_init=init_state, sequence=sequence, usePlotly=False, filename="Readout_population.png")


# %%
print("----------Plotting IQ----------")
plotIQ(exp, sequence, tf.constant(model.ann_opers[1], dtype=tf.complex128))

# %%



