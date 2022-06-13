from operator import le
from typing import List
import numpy as np
from matplotlib import pyplot as plt, colors, cm
import plotly.graph_objects as go
import pylab
from mpl_toolkits.mplot3d import Axes3D

from utilities_functions import getQubitsPopulation
from c3.experiment import Experiment
import c3.utils.tf_utils as tf_utils
import tensorflow as tf
import math


def plotSignal(time, signal, filename=None):
    """
    Plots a time dependent drive signal.

    Parameters
    ----------
    time
        timestamps
    signal
        the function values
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    time = time.flatten()
    signal = signal.flatten()
    plt.plot(time, signal)
    plt.xlabel("Time")
    plt.ylabel("Signal")

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotSignalSpectrum(
    time: np.array,
    signal: np.array,
    spectrum_threshold: float = 1e-4,
    filename: str = None,
    usePlotly=False
):
    """
    Plots the normalised frequency spectrum of a time-dependent signal.

    Parameters
    ----------
    time: np.array
        timestamps
    signal: np.array
        signal value
    spectrum_threshold: float
        If not None, only the part of the normalised spectrum whose absolute square
        is larger than this value will be plotted.
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    # plot time domain
    time = time.flatten()
    signal = signal.flatten()
    plt.figure()

    # calculate frequency spectrum
    freq_signal = np.fft.rfft(signal)
    if np.abs(np.max(freq_signal)) > 1e-14:
        normalised = freq_signal / np.max(freq_signal)
    else:
        normalised = freq_signal
    freq = np.fft.rfftfreq(len(time), time[-1] / len(time))

    # cut spectrum if necessary
    if spectrum_threshold is not None:
        limits = np.flatnonzero(np.abs(normalised) ** 2 > spectrum_threshold)
        freq = freq[limits[0] : limits[-1]]
        normalised = normalised[limits[0] : limits[-1]]

    # plot frequency domain
    if usePlotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=normalised.real, name="Re", mode="lines"))
        fig.add_trace(go.Scatter(x=freq, y=normalised.imag, name="Im", mode="lines"))
        fig.add_trace(go.Scatter(x=freq, y=np.abs(normalised)**2, name="Square", mode="lines"))
        fig.show()

    else:
        plt.plot(freq, normalised.real, label="Re")
        plt.plot(freq, normalised.imag, label="Im")
        plt.plot(freq, np.abs(normalised) ** 2, label="Square")
        plt.xlabel("frequency")
        plt.legend()

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotComplexMatrix(
    M: np.array,
    colourMap: str = "nipy_spectral",
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    filename: str = None,
):
    """
    Plots a complex matrix as a 3d bar plot, where the radius is the bar height and the phase defines
    the bar colour.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : str
      a name of a colormap to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    z1 = np.absolute(M)
    z2 = np.angle(M)

    # mesh
    lx = z1.shape[1]
    ly = z1.shape[0]
    xpos, ypos = np.meshgrid(
        np.arange(0.25, lx + 0.25, 1), np.arange(0.25, ly + 0.25, 1)
    )
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    # bar sizes
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz1 = z1.flatten()
    dz2 = z2.flatten()

    # plot the bars
    fig = plt.figure()
    axis = fig.add_subplot(111, projection="3d")
    colours = cm.get_cmap(colourMap)
    for idx, cur_zpos in enumerate(zpos):
        color = colours((dz2[idx] + np.pi) / (2 * np.pi))
        axis.bar3d(
            xpos[idx],
            ypos[idx],
            cur_zpos,
            dx[idx],
            dy[idx],
            dz1[idx],
            alpha=1,
            color=color,
        )

    # view, ticks and labels
    axis.view_init(elev=30, azim=-15)
    axis.set_xticks(np.arange(0.5, lx + 0.5, 1))
    axis.set_yticks(np.arange(0.5, ly + 0.5, 1))
    if xlabels is not None:
        axis.w_xaxis.set_ticklabels(xlabels, fontsize=13 - 2 * (len(xlabels) / 8))
    if ylabels is not None:
        axis.w_yaxis.set_ticklabels(
            ylabels, fontsize=13 - 2 * (len(ylabels) / 8), rotation=-65
        )

    # colour bar
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=axis,
        shrink=0.6,
        pad=0.1,
        ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
    )
    cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def plotComplexMatrixAbsOrPhase(
    M: np.array,
    colourMap: str = "nipy_spectral",
    xlabels: List[str] = None,
    ylabels: List[str] = None,
    phase: bool = True,
    filename: str = None,
):
    """
    Plots the phase or absolute value of a complex matrix as a 2d colour plot.

    Parameters
    ----------
    M : np.array
      the matrix to plot
    colourMap : str
      name of a colour map to be used for the phases
    xlabels : List[str]
      labels for the x-axis
    ylabels : List[str]
      labels for the y-axis
    phase : bool
      whether the phase or the absolute value should be plotted
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    """
    data = np.angle(M) if phase else np.abs(M)

    # grid
    lx = M.shape[1]
    ly = M.shape[0]
    extent = [0.5, lx + 0.5, 0.5, ly + 0.5]

    # plot
    fig = plt.figure()
    axis = fig.add_subplot(111)
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    colours = cm.get_cmap(colourMap)
    axis.imshow(
        data,
        cmap=colours,
        norm=norm,
        interpolation=None,
        extent=extent,
        aspect="auto",
        origin="lower",
    )

    # ticks and labels
    axis.set_xticks(np.arange(1, lx + 1, 1))
    axis.set_yticks(np.arange(1, ly + 1, 1))
    if xlabels is not None:
        axis.xaxis.set_ticklabels(xlabels, fontsize=12, rotation=-90)
    if ylabels is not None:
        axis.yaxis.set_ticklabels(ylabels, fontsize=12)

    # colour bar
    if phase:
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
        ticks = [-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi]
    else:
        norm = colors.Normalize(vmin=0, vmax=np.max(data))
        ticks = np.linspace(0, np.max(data), 5)
    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=colours),
        ax=axis,
        shrink=0.8,
        pad=0.1,
        ticks=ticks,
    )
    if phase:
        cbar.ax.set_yticklabels(["$-\\pi$", "$\\pi/2$", "0", "$\\pi/2$", "$\\pi$"])

    # show and save
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def calculatePopulation(
    exp: Experiment, psi_init: tf.Tensor, sequence: List[str]
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
    exp: Experiment,
    psi_init: tf.Tensor,
    sequence: List[str],
    labels: List[str] = None,
    states_to_plot: List[tuple] = None,
    usePlotly=True,
    vertical_lines=False,
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
    if usePlotly:
        fig = go.Figure()
        if states_to_plot:
            for i in range(len(pop_t.T[0])):
                if legend_labels[i] in states_to_plot:
                    fig.add_trace(
                        go.Scatter(
                            x=ts / 1e-9,
                            y=pop_t.T[:, i],
                            mode="lines",
                            name=str(legend_labels[i]),
                        )
                    )
        else:
            for i in range(len(pop_t.T[0])):
                fig.add_trace(
                    go.Scatter(
                        x=ts / 1e-9,
                        y=pop_t.T[:, i],
                        mode="lines",
                        name=str(legend_labels[i]),
                    )
                )
        fig.update_layout(xaxis_title=labelX, yaxis_title=labelY)
        fig.update_layout(width = 600, height=400)
    else:
        fig, axs = plt.subplots(1, 1, figsize=[10, 5])
        if states_to_plot:
            for i in range(len(pop_t.T[0])):
                if legend_labels[i] in states_to_plot:
                    axs.plot(ts / 1e-9, pop_t.T[:,i])

            # set plot properties
            axs.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
            axs.set_xlabel(labelX)
            axs.set_ylabel(labelY)
            legends = [i for i in legend_labels if i in states_to_plot]
            plt.legend(
                legends,
                ncol=int(np.ceil(model.tot_dim / 15)),
                bbox_to_anchor=(1.05, 1.0),
                loc="upper left",
            )
            plt.tight_layout()
        else:
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

    # plot vertical lines; TODO: does not work with Plotly yet!
    if (not usePlotly) and vertical_lines and len(sequence) > 0:
        gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
        for i in range(1, len(gate_steps)):
            gate_steps[i] += gate_steps[i - 1]
        gate_times = gate_steps * dt
        if usePlotly:
            for t in gate_times:
                fig.add_vline(
                    x=t / 1e-9, line_width=1, line_dash="dot", line_color="black"
                )
        else:
            plt.vlines(
                x=gate_times / 1e-9,
                ymin=tf.reduce_min(pop_t),
                ymax=tf.reduce_max(pop_t),
                linestyles=":",
                colors="black",
            )

    # show and save
    if usePlotly:
        if filename:
            fig.write_html(filename+".html")
        #fig.show()
    else:
        if filename:
            plt.savefig(filename, bbox_inches="tight", dpi=100)
        plt.show()
        plt.close()


def plotSplittedPopulation(
    exp: Experiment,
    psi_init: tf.Tensor,
    sequence: List[str],
    vertical_lines=False,
    filename: str = None,
) -> None:
    """
    Plots time dependent populations for multiple qubits in separate plots.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: np.array
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state
    vertical_lines: bool
        If true, this add a dotted vertical line after each gate
    filename: str
        Optional name of the file to which the plot will be saved. If none,
        it will only be shown.
    -------
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    pop_t = calculatePopulation(exp, psi_init, sequence)
    dims = [s.hilbert_dim for s in model.subsystems.values()]
    splitted = getQubitsPopulation(pop_t, dims)

    # timestamps
    dt = exp.ts[1] - exp.ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])

    # positions of vertical lines
    gate_steps = [exp.partial_propagators[g].shape[0] for g in sequence]
    for i in range(1, len(gate_steps)):
        gate_steps[i] += gate_steps[i - 1]
    gate_times = gate_steps * dt

    # create both subplots
    fig, axs = plt.subplots(1, len(splitted), sharey="all")
    for idx, ax in enumerate(axs):
        ax.plot(ts / 1e-9, splitted[idx].T)
        if vertical_lines:
            ax.vlines(
                gate_times / 1e-9,
                tf.reduce_min(pop_t),
                tf.reduce_max(pop_t),
                linestyles=":",
                colors="black",
            )

        # set plot properties
        ax.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Population")
        ax.legend([str(x) for x in np.arange(dims[idx])])

    plt.tight_layout()

    # show and save
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close()


def calculateLaguerrePolynomial(
    n: int, 
    R: tf.Tensor
):
    beta = 2*R
    Ln = tf.zeros_like(beta, dtype=tf.double)
    for m in range(n+1):
        Ln += (-(tf.abs(beta)**2))**m * math.comb(n, m)/math.factorial(m)
    Ln = tf.cast(Ln, dtype=tf.complex128)
    return Ln


def calculateStateWignerFunction(
    state: tf.Tensor,
    xvec: np.array,
    yvec: np.array
) -> tf.Tensor:
    
    X, Y = tf.meshgrid(xvec, yvec)
    Wmat = tf.zeros_like(X, dtype=tf.complex128)
    X = tf.cast(X, dtype=tf.complex128)
    Y = tf.cast(Y, dtype=tf.complex128)
    R = X + 1j * Y
    coeff = (2/np.pi) * tf.exp(-2*tf.abs(R)**2)
    coeff = tf.cast(coeff, dtype=tf.complex128)

    for n in range(state.shape[0]):
        Wmat += coeff *(-1)**(n) * calculateLaguerrePolynomial(n, R) * state[n][0]
    
    return Wmat


def plotWignerFunction(
    states: List[tf.Tensor],
    xvec: np.array,
    yvec: np.array
) -> None:
    """
    Plots wigner function of the state.
    An iterative method is used for calculating the winger function.
    This is calculated as W = \sum_n \psi_{n} W_{n}, 
    where \psi_{n} are the elements of state vector and W_{n} are the
    wigner functions corresponding to the state \ket{n}.   
     
    # To-Do - Make wigner function for density matrices as well.

    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    state: tf.tensor
        State or density matrix.
    xvec: np.array
        X-coordinates where to calculate the wigner function.
    yvec: np.array
        Y-coordinates where to calculate the wigner function.
    -------
    """
    X, Y = tf.meshgrid(xvec, yvec)
    Wmat = tf.zeros_like(X, dtype=tf.complex128)
    
    for state in states:
        if state.shape[0] == state.shape[1]:
            print("Not implemented now")
            return 0
        else:
            Wmat += calculateStateWignerFunction(state, xvec, yvec)

    
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(np.real(X), np.real(Y), np.real(Wmat))
    plt.show()
    
    plt.imshow(np.real(Wmat))
    plt.colorbar()
    plt.show()

def calculateState(
    exp: Experiment,
    psi_init: tf.Tensor,
    sequence: List[str]
):

    """
    Calculates the state of system with time. Returns the state for a coherent simulation.
    For a Master equation simulation this returns a list of vectorized density matrices.

    Parameters
    ----------
    exp: Experiment,
        The experiment containing the model and propagators
    psi_init: tf.Tensor,
        Initial state vector or density matrix
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
    psi_list = []
    for gate in sequence:
        for du in dUs[gate]:
            psi_t = tf.matmul(du, psi_t)
            psi_list.append(psi_t)

    return psi_list

def calculateExpectationValue(states, Op, lindbladian):
    """
    Calculates expectation value of the operator Op for the states.
    If lindbladian is True, then states should be vectorized density matrices.

    Args:
        states (tf.Tensor): State vector or density matrices
        Op (tf.Tensor): Operator
        lindbladian (bool): True for Master equation simulation

    Returns:
        expect_val (List[tf.complex128]): Expectation value of operator Op
    """
    expect_val = []
    for i in states:
        if lindbladian:
            expect_val.append(tf.linalg.trace(tf.matmul(i, Op)))
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


def frameOfDrive(exp, psi_list, freq_q, freq_r, spacing, t_total):
    """
    Rotates the states to the frame of drive.

    Args:
        exp (Experiment): Experiment
        psi_list (List[tf.Tensor]): List of states/vectorized density matrices
        freq (float): Frequency of rotating frame
        k (Int): Compute the IQ plane points with a spacing of k points in between.
                 This is just to decrease the computational overhead.

    Returns:
        psi_rotated (List[tf.Tensor]): List of states in the rotated frame.
    """
    
    model = exp.pmap.model
    aR = tf.convert_to_tensor(model.ann_opers[1], dtype = tf.complex128)
    aQ = tf.convert_to_tensor(model.ann_opers[0], dtype = tf.complex128)

    n = len(psi_list)

    aR_dag = tf.transpose(aR, conjugate=True)
    NR = tf.matmul(aR_dag,aR)

    aQ_dag = tf.transpose(aQ, conjugate=True)
    NQ = tf.matmul(aQ_dag, aQ)

    #ts = exp.ts[::spacing]
    ts = tf.linspace(0.0, t_total, n, name="linspace")
    ts = tf.cast(ts, dtype=tf.complex128)
    
    I = tf.eye(len(aR), dtype=tf.complex128)
    
    psi_rotated = []
    
    for i in range(n):
        U = tf.linalg.expm(1j*2*np.pi*(freq_r*NR + freq_q*NQ)*ts[i])
        if model.lindbladian:
            U_dag = tf.transpose(U, conjugate=True)
            rho_i = tf_utils.tf_vec_to_dm(psi_list[i])
            psi_rotated.append(tf.matmul(tf.matmul(U_dag, rho_i), U))    ## Check this. Now it is U_dag*rho*U but I think it should be U*rho*U_dag
        else:
            psi_rotated.append(tf.matmul(U, psi_list[i]))

    return psi_rotated


def plotIQ(
        exp: Experiment, 
        sequence: List[str], 
        annihilation_operator: tf.Tensor,
        drive_freq_q,
        drive_freq_r,
        t_total,
        spacing=100,
        usePlotly=False,
        filename=None,
        connect_points=False,
        second_excited=False
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
    psi_list = psi_list[::spacing]
    psi_list_0 =  frameOfDrive(exp, psi_list, drive_freq_q, drive_freq_r, spacing, t_total)
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
    psi_list = psi_list[::spacing]
    psi_list_1 =  frameOfDrive(exp, psi_list, drive_freq_q, drive_freq_r, spacing, t_total)
    expect_val_1 = calculateExpectationValue(psi_list_1, annihilation_operator, model.lindbladian)
    Q1 = np.real(expect_val_1)
    I1 = np.imag(expect_val_1)


    if second_excited:
        state_index = exp.pmap.model.get_state_index((2,0))
        psi_init_2 = [[0] * model.tot_dim]
        psi_init_2[0][state_index] = 1
        init_state_2 = tf.transpose(tf.constant(psi_init_2, tf.complex128))
        
        if model.lindbladian:
            init_state_2 = tf_utils.tf_state_to_dm(init_state_2)

        psi_list = calculateState(exp, init_state_2, sequence)
        psi_list = psi_list[::spacing]
        psi_list_2 =  frameOfDrive(exp, psi_list, drive_freq_q, drive_freq_r, spacing, t_total)
        expect_val_2 = calculateExpectationValue(psi_list_2, annihilation_operator, model.lindbladian)
        Q2 = np.real(expect_val_2)
        I2 = np.imag(expect_val_2)

    n = len(psi_list)
    ts = tf.linspace(0.0, t_total, n, name="linspace")
    dist = []
    for t in range(len(ts)):
        dist.append(np.sqrt((Q0[t] - Q1[t])**2 + (I0[t] - I1[t])**2))
    
    if usePlotly:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = Q0, y = I0, mode = "lines", name="Ground state"))
        fig.add_trace(go.Scatter(x = Q1, y = I1, mode = "lines", name ="Excited state"))
        if second_excited:
            fig.add_trace(go.Scatter(x = Q2, y = I2, mode = "lines", name ="Second excited state"))
        fig.show()
        if filename:
            fig.write_image(filename+"_Readout_IQ.png")
    else:
        plt.figure(dpi=150)
        plt.plot(Q0, I0, label="Ground state", linestyle='--', marker='o')
        plt.plot(Q1, I1, label="Excited state", linestyle='--', marker='o')
        if second_excited:
            plt.plot(Q2, I2, label="Second excited state", linestyle='--', marker='o')

        
        if connect_points:
            for x1,x2,y1,y2 in zip(Q0, Q1, I0, I1):
                plt.plot([x1, x2], [y1,y2], linestyle="dotted", color="black")

        plt.legend()
        plt.show()
        if filename:
            plt.savefig(filename+"_IQ Plot.png")
    
    plt.figure(dpi=100)
    plt.plot(ts, dist)
    plt.xlabel("Time (in seconds)")
    plt.ylabel("Distance (in arbitrary units)")
    plt.show()
    if filename:
        plt.savefig(filename+"_Distance_plot.png")



def plotPulseShapes(pmap, sequence, drive_channels, pulse_names, ts):
    """
    Plot the pulse shapes from the parameter map.

    Args:
        pmap (ParameterMap): Parameter Map
        sequence (List[str]): List of strings of gates.
        drive_channels (List[str]): List of string of drive channel names.
        pulse_names (List[str]): List of string of names of pulses
        ts (List): Array of time points to plot
    """

    pmap_dict = pmap.asdict()
    pulse_list = []
    pulse_shapes = []
    for gate in sequence:
        for drive, pulse in zip(drive_channels, pulse_names):
            pulse_list.append(pmap_dict[gate]["drive_channels"][drive][pulse])

    for pulse in pulse_list:
        pulse_shapes.append(pulse.shape(ts, pulse.params) * pulse.params["amp"])


    plt.figure(dpi=100)
    for i in range(len(pulse_shapes)):
        plt.plot(ts, pulse_shapes[i], label=drive_channels[i])

    plt.legend()


def plotPopulationFromState(
    exp: Experiment,
    init_state: tf.Tensor,
    sequence: List[str],
    Num_shots = 1
):

    model = exp.pmap.model
    if model.lindbladian:
        result = exp.solve_lindblad_ode(init_state, sequence)
        rhos = result["states"]
        ts = result["ts"]
        pops = []
        for rho in rhos:
            pops.append(tf.math.real(tf.linalg.diag_part(rho)))
        plt.figure(figsize=[10,5])
        plt.plot(ts, pops)
        plt.legend(
            model.state_labels,
            ncol=int(np.ceil(model.tot_dim / 15)),
            bbox_to_anchor=(1.05, 1.0),
            loc="upper left")
        plt.tight_layout()
    else:
        result = exp.solve_stochastic_ode(init_state, sequence, Num_shots)
        psis = result["states"]
        ts = result["ts"]
        pops = []
        for i in range(Num_shots):
            pops_shots = []
            for psi in psis[i]:
                pop_t = tf.abs(psi)**2
                pops_shots.append(tf.reshape(pop_t, pop_t.shape[:-1]))
            pops.append(pops_shots)

        plt.figure(figsize=[10,5])
        for i in range(len(pops)):
            plt.plot(ts, pops[i])
            plt.legend(
                model.state_labels,
                ncol=int(np.ceil(model.tot_dim / 15)),
                bbox_to_anchor=(1.05, 1.0),
                loc="upper left")
            plt.tight_layout()