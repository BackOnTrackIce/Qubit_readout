import itertools
from typing import List, Tuple

import numpy as np
import copy

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.generator.generator import Generator as Gnr


# Building blocks
import c3.libraries.chip as chip
import c3.generator.devices as devices
import c3.signal.pulse as pulse
import c3.signal.gates as gates

# Libs and helpers
import c3.libraries.hamiltonians as hamiltonians


def createQubits(
    qubit_levels_list: List[int],
    freq_list: List[float],
    anharm_list: List[float],
    t1_list: List[float],
    t2star_list: List[float],
    phi_list: List[float],
    phi0_list: List[float],
    d_list: List[float],
    qubit_temp: float,
) -> List[chip.Transmon]:
    """
    Creates and returns a list of qubits.

    Parameters
    ----------
    qubit_levels_list: List[int]
        List of number of levels in each qubit.
    freq_list: List[Float]
        List of frequency of each qubit.
    anharm_list: List[Float]
        List of anharmonicity of each qubit.
    t1_list: List[Float]
        List of T1 values for each qubit.
    t2star_list: List[Float]
        List of T2* values for each qubit.
    phi_list: List[Float]
        List of current flux values for each qubit.
    phi0_list: List[Float]
        List of flux bias for each qubit.
    d_list: List[Float]
        List of junction asymmetry values for each qubit.
    qubit_temp: Float
        Temperature of the qubits

    Returns
    -------
    qubits: List[chip.Transmon]
        An list of transmons
    """

    qubits = []
    for i in range(len(qubit_levels_list)):
        qubits.append(
            chip.Transmon(
                name=f"Q{i + 1}",
                desc=f"Qubit {i + 1}",
                freq=Qty(
                    value=freq_list[i],
                    min_val=freq_list[i] - 5e6,
                    max_val=freq_list[i] + 5e6,
                    unit="Hz 2pi",
                ),
                anhar=Qty(
                    value=anharm_list[i], min_val=-380e6, max_val=-120e6, unit="Hz 2pi"
                ),
                hilbert_dim=qubit_levels_list[i],
                t1=Qty(value=t1_list[i], min_val=1e-6, max_val=90e-6, unit="s"),
                t2star=Qty(
                    value=t2star_list[i], min_val=10e-6, max_val=90e-6, unit="s"
                ),
                temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
                phi=Qty(value=phi_list[i], max_val=5.0, min_val=0.0, unit="Wb"),
                phi_0=Qty(value=phi0_list[i], max_val=11.0, min_val=9.0, unit="Wb"),
                d=Qty(value=d_list[i], max_val=0.1, min_val=-0.1, unit=""),
            )
        )

    return qubits


def createChainCouplings(
    coupling_strength: List[float],
    qubits: List[chip.Transmon],
) -> List[chip.Coupling]:
    """
    Creates nearest neighbour couplings between the qubits in 1D chain. This directly couples the qubits without
    intermediate couplers and assumes a linear chain architecture of qubits.

    Parameters
    ----------
    coupling_strength: List[float]
        List of coupling strength between adjacent Qubits. For n qubits, this needs to be a list of n-1 values.
    qubits: List[chip.Transmon]
        List of qubits

    Returns
    -------
    List of couplings
    """
    if len(coupling_strength) < len(qubits) - 1:
        raise Exception("not enough coupling constants")

    couplings = []

    for i in range(len(qubits) - 1):
        couplings.append(
            chip.Coupling(
                name=f"{qubits[i].name}-{qubits[i + 1].name}",
                desc="Coupling",
                comment=f"Coupling between {qubits[i].name} and {qubits[i + 1].name}",
                connected=[qubits[i].name, qubits[i + 1].name],
                strength=Qty(
                    value=coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e6,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    return couplings


def createChainCouplingsWithCouplers(
    coupling_strength: List[float],
    qubits: List[chip.Transmon],
    couplers: List[chip.Transmon],
) -> List[chip.Coupling]:
    """
    Creates and returns a list of Nearest neighbour (NN) couplings where each pair of qubits is coupled via a coupler.
    This assumes a linear chain architecture of Qubits and couplers.

    Parameters
    ----------
    coupling_strength: List[float]
        List of coupling strength between the Qubits and couplers. For n qubits, this list needs to contain 2(n-1)
        values, two for each coupler.
    qubits: List[chip.Transmon]
        List of qubits
    couplers: List[chip.Transmon]
        List of couplers. For n qubits, this list needs to contain n-1 couplers.

    Returns
    -------
    List of couplings
    """
    if len(couplers) < len(qubits) - 1:
        raise Exception("not enough couplers")
    if len(coupling_strength) < 2 * len(couplers):
        raise Exception("not enough coupling constants")

    g_NN_array = []

    for i in range(2 * len(couplers)):
        coupler_index = int(np.floor(i / 2))
        qubit_index = int(np.floor((i + 1) / 2))
        g_NN_array.append(
            chip.Coupling(
                name=f"{qubits[qubit_index].name}-{couplers[coupler_index].name}",
                desc="Coupling",
                comment=f"Coupling between {qubits[qubit_index].name} and {couplers[coupler_index].name}",
                connected=[
                    qubits[qubit_index].name,
                    couplers[coupler_index].name,
                ],
                strength=Qty(
                    value=coupling_strength[i],
                    min_val=-1 * 1e3,
                    max_val=200e9,
                    unit="Hz 2pi",
                ),
                hamiltonian_func=hamiltonians.int_XX,
            )
        )

    return g_NN_array


def createDrives(qubits: List[chip.Transmon]) -> List[chip.Drive]:
    """
    Creates and returns a drive line for each qubit in the list.

    Parameters
    ----------
    qubits : List[chip.PhysicalComponent]
        One drive line is generated for each qubit in this list. The qubits are not altered.

    Returns
    -------
    List[chip.Drive]
        The drive lines in the same order as the list of qubits.
    """
    drives = []
    for i in range(len(qubits)):
        drives.append(
            chip.Drive(
                name=f"d{i+1}",
                desc=f"Drive {i+1}",
                comment=f"Drive line on qubit {qubits[i].name}",
                connected=[qubits[i].name],
                hamiltonian_func=hamiltonians.x_drive,
            )
        )
    return drives


def createGenerator(
    drives: List[chip.Drive],
    sim_res: float = 100e9,
    awg_res: float = 2e9,
) -> Gnr:
    """
    Creates and returns the generator.

    Parameters
    ----------
    drives: List[chip.Drive]
        List of drives on the Qubits
    sim_res: float
        Resolution of the simulation
    awg_res: float
        Resolution of AWG

    Returns
    -------
    Generator
    """
    chain = ["LO", "AWG", "DigitalToAnalog", "Response", "Mixer", "VoltsToHertz"]
    chains = {f"{d.name}": chain for d in drives}

    generator = Gnr(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Response": devices.Response(
                name="resp",
                rise_time=Qty(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="V_to_Hz",
                V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains=chains,
    )

    return generator


def createCarriers(qubit_freqs: List[float], sideband: float) -> List[pulse.Carrier]:
    """
    Creates and returns a carrier for each qubit.

    Parameters
    ----------
    qubit_freqs: List[float]
        List of frequencies of the qubits.
        For tunable qubits, one can use model.get_qubit_freqs()
        to find the frequency of the qubits at current flux.
    sideband: float
        Frequency of the sideband.

    Returns
    -------
    List of carriers for each qubit
    """
    carrier_array = []

    for i in range(len(qubit_freqs)):
        carrier_parameters = {
            "freq": Qty(
                value=qubit_freqs[i] + sideband, min_val=1e8, max_val=8e9, unit="Hz 2pi"
            ),
            "framechange": Qty(
                value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"
            ),
        }
        carrier_array.append(
            pulse.Carrier(
                name="carrier",
                desc="Frequency of the local oscillator",
                params=carrier_parameters,
            )
        )

    return carrier_array


def createSingleQubitGate(
    name: str,
    drives: List[chip.Drive],
    carriers: List[pulse.Carrier],
    target: int,
    t_final: float,
    target_pulse: pulse.Envelope,
    non_target_pulse: pulse.Envelope,
    sideband: float,
    xy_angle: float = None,
) -> gates.Instruction:
    """
    Creates an instruction that represents a single qubit gate. This applies the target pulse to the target qubit's
    drive line and the "non-target pulse" to all other qubits.

    Parameters
    ----------
    name: str
        Name of the gate
    drives: List[chip.Drive]
        List of drive lines for each qubit
    carriers: List[pulse.Carrier]
        list of carrier pulses for each qubit
    target: int
        index of the target qubit on which the gate should act
    t_final: float
        Final gate time for single qubit rotation gates
    target_pulse: pulse.Envelope
        pulse for single qubit rotation gates
    non_target_pulse: pulse.Envelope
        pulse for correcting phase
    sideband: float
        Frequency of sideband
    xy_angle: float
        The angle by which the target's drive should be shifted to create a different gate

    Returns
    -------
    Single qubit rotation gates for one qubit on the chip.
    """
    gate = gates.Instruction(
        name=name,
        targets=[target],
        t_start=0.0,
        t_end=t_final,
        channels=[d.name for d in drives],
    )
    for i in range(len(drives)):
        if i == target:
            gate.add_component(copy.deepcopy(target_pulse), drives[i].name)
            gate.add_component(copy.deepcopy(carriers[i]), drives[i].name)
            if xy_angle is not None:
                gate.comps[drives[i].name][target_pulse.name].params[
                    "xy_angle"
                ].set_value(xy_angle)
        else:
            carrier = copy.deepcopy(carriers[i])
            carrier.params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
            )
            gate.add_component(copy.deepcopy(non_target_pulse), drives[i].name)
            gate.add_component(carrier, drives[i].name)

    return gate


def createTwoQubitsGate(
    name: str,
    drives: List[chip.Drive],
    carriers: List[pulse.Carrier],
    targets: List[int],
    t_final: float,
    target_pulses: Tuple[pulse.Envelope, pulse.Envelope],
    non_target_pulse: pulse.Envelope,
    sideband: float,
    ideal: np.array = None,
) -> gates.Instruction:
    """
    Creates a gate that acts on a list of qubits. This applies a copy of the target pulse to all the target qubit's
    drive lines and the "non-target pulse" to all other qubits.

    Parameters
    ----------
    name: str
        Name of the gate
    drives: List[chip.Drive]
        List of drive lines for each qubit
    carriers: List[pulse.Carrier]
        list of carrier pulses for each qubit
    targets: List[int]
        indices of the target qubits on which the gate should act
    t_final: float
        Final gate time for single qubit rotation gates
    target_pulses: Tuple[pulse.Envelope, pulse.Envelope]
        pulses for both target drives in two qubit gates
    non_target_pulse: pulse.Envelope
        pulse for correcting phase
    sideband: float
        Frequency of sideband
    ideal: np.array
        Matrix representation of the ideal gate, if not specified by the name.

    Returns
    -------
    Single qubit rotation gates for one qubit on the chip.
    """
    gate = gates.Instruction(
        name=name,
        targets=targets,
        t_start=0.0,
        t_end=t_final,
        channels=[d.name for d in drives],
        ideal=ideal,
    )

    for i in range(len(drives)):
        if i in targets:
            idx = targets.index(i)
            gate.add_component(target_pulses[idx], drives[i].name)
            gate.add_component(copy.deepcopy(carriers[i]), drives[i].name)
        else:
            carrier = copy.deepcopy(carriers[i])
            carrier.params["framechange"].set_value(
                (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
            )
            gate.add_component(copy.deepcopy(non_target_pulse), drives[i].name)
            gate.add_component(carrier, drives[i].name)

    return gate


def getQubitsPopulation(population: np.array, dims: List[int]) -> np.array:
    """
    Splits the population of all levels of a system into the populations of levels per subsystem.
    Parameters
    ----------
    population: np.array
        The time dependent population of each energy level. First dimension: level index, second dimension: time.
    dims: List[int]
        The number of levels for each subsystem.
    Returns
    -------
    np.array
        The time-dependent population of energy levels for each subsystem. First dimension: subsystem index, second
        dimension: level index, third dimension: time.
    """
    numQubits = len(dims)

    # create a list of all levels
    qubit_levels = []
    for dim in dims:
        qubit_levels.append(list(range(dim)))
    combined_levels = list(itertools.product(*qubit_levels))

    # calculate populations
    qubitsPopulations = np.zeros((numQubits, dims[0], population.shape[1]))
    for idx, levels in enumerate(combined_levels):
        for i in range(numQubits):
            qubitsPopulations[i, levels[i]] += population[idx]
    return qubitsPopulations
