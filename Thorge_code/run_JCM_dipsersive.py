import numpy as np
import qutip as qp
from qutip.tensor import tensor
from qutip.states import fock, destroy, coherent
from qutip.operators import qeye, sigmaz, sigmax, sigmay, sigmam, sigmap
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import pandas as pd
import multiprocessing
import os

def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

class setup_Atom_Cavity:
    def __init__(self, drive_frequenz, resonator_frequenz, qubit_frequenz, drive_ampltidue, interaction_strength):
        self.drive_frequenz = drive_frequenz
        self.resonator_frequenz = resonator_frequenz
        self.qubit_frequenz = qubit_frequenz
        self.drive_ampltidue = drive_ampltidue
        self.interaction_strength = interaction_strength
        self.n_crit = self.calc_ncrit()
        self.lamda = self.calc_lamda()
        self.xi = self.calc_xi()
        self.detuning = self.calc_detuning()

    def calc_lamda(self):
        print('lamda << 1: ', ((self.interaction_strength) / (self.qubit_frequenz - self.resonator_frequenz)))
        return ((self.interaction_strength) / (self.qubit_frequenz - self.resonator_frequenz))

    def calc_ncrit(self):
        print('n_crit: ', (((self.qubit_frequenz - self.resonator_frequenz) * (
                self.qubit_frequenz - self.resonator_frequenz)) / (
                                   4 * self.interaction_strength * self.interaction_strength)))
        return (((self.qubit_frequenz - self.resonator_frequenz) * (self.qubit_frequenz - self.resonator_frequenz)) / (
                4 * self.interaction_strength * self.interaction_strength))

    def calc_xi(self):
        print('xi: ',
              self.interaction_strength * self.interaction_strength / (self.qubit_frequenz - self.resonator_frequenz))
        return self.interaction_strength * self.interaction_strength / (self.qubit_frequenz - self.resonator_frequenz)

    def calc_detuning(self):
        return self.qubit_frequenz - self.resonator_frequenz


class setup_Dispersive_readout:
    def __init__(self, cavity_space, cavity_photon_number, times, option, parameters, state_selected,system_selected,mode_2pi):
        self.mode_2pi = mode_2pi
        if self.mode_2pi == True:
            self.drive_frequenz = parameters.drive_frequenz/ 2*np.pi
            self.resonator_frequenz = parameters.resonator_frequenz/ 2*np.pi
            self.qubit_frequenz = parameters.qubit_frequenz/ 2*np.pi
            self.drive_ampltidue = parameters.drive_ampltidue/ 2*np.pi
            self.interaction_strength = parameters.interaction_strength/ 2*np.pi
        else:
            self.drive_frequenz = parameters.drive_frequenz
            self.resonator_frequenz = parameters.resonator_frequenz
            self.qubit_frequenz = parameters.qubit_frequenz
            self.drive_ampltidue = parameters.drive_ampltidue
            self.interaction_strength = parameters.interaction_strength

        self.drive_frequenz = parameters.drive_frequenz
        self.resonator_frequenz = parameters.resonator_frequenz
        self.qubit_frequenz = parameters.qubit_frequenz
        self.drive_ampltidue = parameters.drive_ampltidue
        self.interaction_strength = parameters.interaction_strength
        self.lamda = parameters.lamda
        self.xi = parameters.xi
        self.detuning = parameters.detuning
        self.cavity_space = cavity_space
        self.state_selected = state_selected
        self.cavity_photon_number = cavity_photon_number

        self.cavity = tensor(qeye(2), destroy(cavity_space))
        self.qubit = tensor(sigmaz(), qeye(cavity_space))
        self.options = option
        self.state_to_calculate()

        self.transform_to_SW = False
        self.build_SW_Transformaton()
        self.build_SW_Operators()
        self.set_I_Q()

        self.times = times

        #if system_selected == 'Disp':
        #    self.run_Disspersive()
        #elif system_selected == 'JCM':
        #    self.run_JCM()



    def set_states_and_times_from_mesolve(self,path):
        self.files_list = os.listdir(path)
        self.times_from_data = []
        self.states_from_data = []

        for i in self.files_list:
            dataset = pd.read_pickle(path + i)
            self.times_from_data.append(np.array(dataset['time']))
            self.states_from_data.append(np.array(dataset['state']))



    def state_to_calculate(self):
        if self.state_selected=="plus_state":
         self.psi = tensor((1 / (np.sqrt(2))) * (qp.basis(2, 0) + qp.basis(2, 1)),fock(self.cavity_space, self.cavity_photon_number))
         self.psi_sw = tensor((1 / (np.sqrt(2))) * (qp.basis(2, 0) + qp.basis(2, 1)), fock(self.cavity_space, self.cavity_photon_number))

        if self.state_selected=='state_up':
          self.psi = tensor(qp.basis(2, 1),fock(self.cavity_space, self.cavity_photon_number))
          self.psi_sw = tensor(qp.basis(2, 1),fock(self.cavity_space, self.cavity_photon_number))


        elif self.state_selected == 'state_down':
             self.psi = tensor(qp.basis(2, 0),fock(self.cavity_space, self.cavity_photon_number))
             self.psi_sw = tensor(qp.basis(2, 0),fock(self.cavity_space, self.cavity_photon_number))

    def build_SW_Transformaton(self):
        SW_Commu = self.lamda * (
                self.cavity * tensor(sigmap(), qeye(self.cavity_space)) - tensor(sigmam(), qeye(
            self.cavity_space)) * self.cavity.dag())
        self.SW = SW_Commu.expm()

    def set_I_Q(self):
        self.I_SW = 0.5 * (self.cavity_sw + self.cavity_sw)
        self.Q_SW = 0.5j * (self.cavity_sw.dag() - self.cavity_sw)
        self.I = 0.5 * (self.cavity + self.cavity)
        self.Q = 0.5j * (self.cavity.dag() - self.cavity)

    def build_SW_Operators(self):
        self.cavity_sw = self.cavity + self.lamda * tensor(sigmam(), qeye(self.cavity_space)) + 0.5 * self.lamda**2 * self.cavity * self.qubit
        self.qubit_sw = self.qubit - 2 * self.lamda * (self.cavity * tensor(sigmap(), qeye(self.cavity_space)) + self.cavity.dag() * tensor(sigmam(), qeye(self.cavity_space))) - self.lamda**2 * self.qubit *(1+2*self.cavity.dag()*self.cavity)

        self.psi.transform(self.SW)

    def build_drive_Transfromation(self, time):
        Drive_frame = self.drive_frequenz * 1j * time * (self.cavity.dag() * self.cavity + self.qubit)
        return Drive_frame.expm()

    def drive_cavity_1(self, time, args):
        return (np.tanh((time - (10 ** -6 / 2 * np.pi)) / (10 ** -6 / 2 * np.pi)) + 1) * (
                0.5 * self.drive_ampltidue * np.exp(-1j * time * self.drive_frequenz))

    def drive_cavity_2(self, time, args):
        return (np.tanh((time - (10 ** -6 / 2 * np.pi)) / (10 ** -6 / 2 * np.pi)) + 1) * (
                0.5 * self.drive_ampltidue * np.exp(1j * time * self.drive_frequenz))

    def build_Hamiltonian_disp(self):
        return ((self.resonator_frequenz + self.xi * self.qubit) * self.cavity.dag() * self.cavity + 0.5 * (
                self.qubit_frequenz + self.xi) * self.qubit)

    def build_JCM(self):
        return (self.resonator_frequenz * self.cavity.dag() * self.cavity +
                self.qubit_frequenz * self.qubit +
                self.interaction_strength * (
                        self.cavity * tensor(sigmap(), qeye(self.cavity_space)) + self.cavity.dag() * tensor(
                    sigmam(), qeye(self.cavity_space))))

    def calculating_trace_difference_JCM_Dispersive(self):
        rho_traceOut_qubit = []
        trace_distance = []
        for times2, state2 in zip (self.times_from_data,self.states_from_data):
            U_exp = [self.build_drive_Transfromation(t) for t in times2]
            state_in_rot_frame = [U_exp[i] * state2[i] * U_exp[i].dag() for i in range(len(times2))]
            rho_traceOut_qubit.append([i.ptrace(1) for i in state_in_rot_frame])

        for rhos in grouped(rho_traceOut_qubit,2):
            trace_distance.append([qp.tracedist(rho_traceOut_qubit[0][i], rho_traceOut_qubit[1][i]) for i in range(len(self.times_from_data[0]))])

        plt.figure(1)
        for tr_distance in trace_distance:
            plt.plot(self.times_from_data[0], tr_distance)
        plt.ylabel('Trace Distance of cavity')
        plt.xlabel('times')
        plt.show()

    def calculating_expectation_values_IQ(self, System):

        print('done with U transformation after the expectation value')
        Q_values_state = []
        I_values_state = []
        for time2, state2, sytsem_type in zip(self.times_from_data, self.states_from_data,System):
            if sytsem_type == 'Disp':
                I = self.I_SW
                Q = self.Q_SW
            else:
                I = self.I
                Q = self.Q
            U_exp = [self.build_drive_Transfromation(t) for t in time2]
            state_in_rot_frame = [U_exp[i] * state2[i] * U_exp[i].dag() for i in range(len(time2))]
            Q_values_state.append([qp.expect(I, i) for i in state_in_rot_frame])
            I_values_state.append([qp.expect(Q, i) for i in state_in_rot_frame])

        plt.figure(1)
        for I_values, Q_values,sytsem_type in zip ( I_values_state,Q_values_state,System):
            plt.plot(I_values,Q_values,label=sytsem_type)
        plt.xlabel('Re <a>')
        plt.ylabel('Im <a>')
        #plt.xlim([-1, 0])
        #plt.ylim([-0.5, 0.5])
        plt.legend()
        plt.show()

    def calculating_expectation_values_occupation(self,System):
        occupation_number = []
        print('done with U transformation after the expectation value')
        for time2, state2,system_type in zip(self.times_from_data, self.states_from_data,System):
            if system_type == 'Disp':
                occupation_Operator = self.cavity_sw.dag() * self.cavity_sw
            else:
                occupation_Operator = self.cavity.dag() * self.cavity
            U_exp = [self.build_drive_Transfromation(t) for t in time2]
            state_in_rot_frame = [U_exp[i] * state2[i] * U_exp[i].dag() for i in range(len(time2))]
            occupation_number.append([qp.expect(occupation_Operator, i) for i in state_in_rot_frame])

        plt.figure(1)
        for times2, occupation_number2, system_type  in zip(self.times_from_data, occupation_number, System):
            plt.plot(times2, occupation_number2, label = system_type)
        plt.xlabel('times')
        plt.ylabel('occupation number')
        plt.legend()
        plt.show()



    def build_drive_Hamiltonian_SW(self):
        return [self.cavity_sw.dag(), self.drive_cavity_1], [self.cavity_sw, self.drive_cavity_2]

    def build_drive_Hamiltonian(self):
        return [self.cavity.dag(), self.drive_cavity_1], [self.cavity, self.drive_cavity_2]

    def run_Disspersive(self):
        data_disp = pd.DataFrame(columns=['time', 'state'])
        print('run dispersive')
        drive = self.build_drive_Hamiltonian_SW()
        Hamiltonian_with_drive = [self.build_Hamiltonian_disp(), drive[0], drive[1]]
        start = time.time()
        data_psi = qp.mesolve(Hamiltonian_with_drive, self.psi_sw, self.times,c_ops=[np.sqrt(0) * self.cavity_sw], e_ops=[], options=self.options)
        self.states_Disp = data_psi.states
        stop = time.time()
        for index, value in enumerate(self.states_Disp):
            results_dict = {}
            results_dict['time'] = self.times[index]
            results_dict['state'] = value
            data_disp = data_disp.append(results_dict, ignore_index=True)

        data_disp.to_pickle('./Disp_'+'_'+str(self.drive_ampltidue/np.power(10, 6))+'_'+self.state_selected+'_'+str(self.mode_2pi)+'_cpu01.pkl')

        print('done Dispersive ', (stop - start) / 60)

    def run_JCM(self):
        data_JCM = pd.DataFrame(columns=['time', 'state'])
        drive = self.build_drive_Hamiltonian()
        Hamiltonian_with_drive = [self.build_JCM(), drive[0], drive[1]]
        print('run JCM')
        start = time.time()
        data_psi = qp.mesolve(Hamiltonian_with_drive, self.psi, self.times,c_ops=[np.sqrt(0) * self.cavity], e_ops=[], options=self.options)
        self.states_JCM= data_psi.states
        stop = time.time()
        for index, value in enumerate(self.states_JCM):
            data_JCM.append({'time': self.times[index], 'state': value}, ignore_index=True)

        data_JCM.to_pickle('./JCM_'+'_'+str(self.drive_ampltidue/np.power(10, 6))+'_'+self.state_selected+'_'+str(self.mode_2pi)+'_cpu01.pkl')
        print('done JCM ', (stop - start) / 60)


def overallFunction():
    jobs = []
    for conf in config_list:
        for drive in drive_amplitude:
            p = multiprocessing.Process(target=runner_JCM_Disp, args=(conf, drive))
            jobs.append(p)
            p.start()

            if len(jobs) == 50:
                for p in jobs:
                    p.join()
                print('waited for 2 in')
                jobs = []


def runner_JCM_Disp(config, drive_ampltidue):
    Atom_cavity = setup_Atom_Cavity(drive_frequenz=3 * np.power(10, 9) * 2 * np.pi,
                                    resonator_frequenz=3 * np.power(10, 9) * 2 * np.pi,
                                    qubit_frequenz=5 * np.power(10, 9) * 2 * np.pi,
                                    drive_ampltidue=10 * np.power(10, 6) * 2 * np.pi,
                                    interaction_strength=drive_ampltidue)

    times = np.linspace(0.0, 0.0000075, 1000)
    option = qp.Options(nsteps=30000)
    test = setup_Dispersive_readout(50, 0, times, option, Atom_cavity, config[1], config[0], config[2])


Atom_cavity = setup_Atom_Cavity(drive_frequenz=3 * np.power(10, 9) * 2 * np.pi,
                                resonator_frequenz=3 * np.power(10, 9) * 2 * np.pi,
                                qubit_frequenz=5 * np.power(10, 9) * 2 * np.pi,
                                drive_ampltidue=10 * np.power(10, 6) * 2 * np.pi,
                                interaction_strength=50 * np.power(10, 6) * 2 * np.pi)


times = np.linspace(0.0, 0.0000002, 100)
option = qp.Options(nsteps=20)
test = setup_Dispersive_readout(20, 0, times, option, Atom_cavity, 'state_up','Disp',False)




System = ['Disp','JCM']
test.set_states_and_times_from_mesolve('./Data/Plot_xTime_yOccupationNumber/JCM_VS_Disp/')
test.calculating_expectation_values_occupation(System)


#def runner_JCM_Disp(config,drive_ampltidue):
#    Atom_cavity = setup_Atom_Cavity(drive_frequenz=3 * np.power(10, 9) * 2 * np.pi,
#                                    resonator_frequenz=3 * np.power(10, 9) * 2 * np.pi,
#                                    qubit_frequenz=5 * np.power(10, 9) * 2 * np.pi,
#                                    drive_ampltidue=drive_ampltidue,
#                                    interaction_strength=50 * np.power(10, 6) * 2 * np.pi)

    #times = np.linspace(0.0, 0.0000075, 1000)
#    times = np.linspace(0.0, 0.0000002, 100)
#    option = qp.Options(nsteps=8000)
#    test = setup_Dispersive_readout(20, 0, times, option, Atom_cavity, config[1], config[0], config[2])


