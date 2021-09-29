import numpy as np
import qutip as qp
from qutip.tensor import tensor
from qutip.states import fock, destroy, coherent
from qutip.operators import qeye, sigmaz, sigmax,sigmay, sigmam, sigmap
import matplotlib.pyplot as plt
import tensorflow as tf



class setup_Atom_Cavity:
    def __init__(self,drive_frequenz,resonator_frequenz,qubit_frequenz,drive_ampltidue,interaction_strength):

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
       print('lamda << 1: ',(( self.interaction_strength) / (self.qubit_frequenz  - self.resonator_frequenz)))
       return (( self.interaction_strength) / (self.qubit_frequenz  - self.resonator_frequenz))


    def calc_ncrit(self):
        print('n_crit: ', (((self.qubit_frequenz - self.resonator_frequenz) * (self.qubit_frequenz - self.resonator_frequenz)) / (4 * self.interaction_strength * self.interaction_strength)))
        return (((self.qubit_frequenz - self.resonator_frequenz) * (self.qubit_frequenz - self.resonator_frequenz)) / (4 * self.interaction_strength * self.interaction_strength))

    def calc_xi(self):
        print('xi: ', self.interaction_strength * self.interaction_strength / (self.qubit_frequenz - self.resonator_frequenz))
        return self.interaction_strength * self.interaction_strength / (self.qubit_frequenz - self.resonator_frequenz)

    def calc_detuning(self):
        return self.qubit_frequenz - self.resonator_frequenz


class setup_Dispersive_readout:
    def __init__(self,cavity_space,cavity_photon_number,time,parameters):

        self.drive_frequenz = parameters.drive_frequenz
        self.resonator_frequenz = parameters.resonator_frequenz
        self.qubit_frequenz = parameters.qubit_frequenz
        self.drive_ampltidue = parameters.drive_ampltidue
        self.interaction_strength = parameters.interaction_strength
        self.lamda = parameters.lamda
        self.xi = parameters.xi
        self.detuning = parameters.detuning
        self.cavity_space = cavity_space

        self.cavity = tensor(qeye(2),destroy(cavity_space))
        self.qubit = tensor(sigmaz(),qeye(cavity_space))

        self.psi1 = tensor(qp.basis(2, 1), fock(self.cavity_space, cavity_photon_number))
        self.psi2 = tensor(qp.basis(2, 0), fock(self.cavity_space, cavity_photon_number))


        self.psi1_sw = tensor(qp.basis(2, 1), fock(self.cavity_space, cavity_photon_number))
        self.psi2_sw = tensor(qp.basis(2, 0), fock(self.cavity_space, cavity_photon_number))


        self.transform_to_SW = False
        self.build_SW_Transformaton()
        self.build_SW_Operators()
        self.set_I_Q()



        self.times = time


    def build_SW_Transformaton(self):
        SW_Commu = self.lamda * (
                    self.cavity * tensor(sigmap(), qeye(self.cavity_space)) - tensor(sigmam(), qeye(self.cavity_space)) * self.cavity.dag())
        self.SW = SW_Commu.expm()

    def set_I_Q(self):
        self.I_SW = 0.5 *    (self.cavity_sw + self.cavity_sw)
        self.Q_SW = 0.5j * (self.cavity_sw.dag() - self.cavity_sw)
        self.I = 0.5 *    (self.cavity + self.cavity)
        self.Q = 0.5j * (self.cavity.dag() - self.cavity)


    def build_SW_Operators(self):
        self.cavity_sw = tensor(qeye(2),destroy(self.cavity_space))
        self.qubit_sw = tensor(sigmaz(),qeye(self.cavity_space))

        self.cavity_sw.transform(self.SW)
        self.qubit_sw.transform(self.SW)
        self.psi1_sw.transform(self.SW)
        self.psi2_sw.transform(self.SW)

    def build_drive_Transfromation(self,time):
        Drive_frame = self.drive_frequenz*1j*time*(self.cavity.dag()*self.cavity+self.qubit)
        return Drive_frame.expm()

    def drive_cavity_1(self,time, args):
        return (np.tanh((time - (10**-6 / 2 * np.pi) ) / (10**-6 / 2 * np.pi) ) + 1)*(0.5*self.drive_ampltidue*np.exp(-1j * time * self.drive_frequenz)) #

    def drive_cavity_2(self,time, args):
        return (np.tanh((time - (10**-6 / 2 * np.pi) ) / (10**-6 / 2 * np.pi) ) + 1)*(0.5*self.drive_ampltidue*np.exp(1j * time * self.drive_frequenz))#(np.tanh((time - (10**-6 / 2 * np.pi)) / (10**-6 / 2 * np.pi) ) + 1)*0.5*


    def build_Hamiltonian_disp(self):
        return((self.resonator_frequenz + self.xi * self.qubit)*self.cavity.dag()*self.cavity + 0.5*(self.qubit_frequenz+self.xi) * self.qubit)


    def build_JCM(self):
        return (self.resonator_frequenz  * self.cavity.dag()*self.cavity +
                self.qubit_frequenz  * self.qubit +
               self.interaction_strength * (self.cavity*tensor(sigmap(),qeye(self.cavity_space))+self.cavity.dag()*tensor(sigmam(),qeye(self.cavity_space))))



    def calculating_expectation_values(self,System):
      if System == 'Disp':
        print('run expectation values for Disp')
        U_exp = [self.build_drive_Transfromation(t) for t in self.times]

        state1_in_rot_frame = [U_exp[i] * self.states_Disp_1[i] * U_exp[i].dag() for i in range(len(self.times))]
        state2_in_rot_frame = [U_exp[i] * self.states_Disp_2[i] * U_exp[i].dag() for i in range(len(self.times))]

        print('done with U transformation after the expectation value')
        self.Q_values_disp_state1 = [qp.expect(self.I_SW,i) for i in state1_in_rot_frame]
        self.I_values_disp_state1 = [qp.expect(self.Q_SW, i) for i in state1_in_rot_frame]

        self.Q_values_disp_state2 = [qp.expect(self.I_SW,i) for i in state2_in_rot_frame]
        self.I_values_disp_state2 = [qp.expect(self.Q_SW, i) for i in state2_in_rot_frame]
        print('Done with expectation values for Disp')

      elif System == 'JCM':
          U_exp = [self.build_drive_Transfromation(t) for t in self.times]

          state1_in_rot_frame = [U_exp[i] * self.states_JCM_1[i] * U_exp[i].dag() for i in range(len(self.times))]
          state2_in_rot_frame = [U_exp[i] * self.states_JCM_2[i] * U_exp[i].dag() for i in range(len(self.times))]

          self.Q_values_JCM_state1 = [qp.expect(self.I, i) for i in state1_in_rot_frame]
          self.I_values_JCM_state1 = [qp.expect(self.Q, i) for i in state1_in_rot_frame]

          self.Q_values_JCM_state2 = [qp.expect(self.I, i) for i in state2_in_rot_frame]
          self.I_values_JCM_state2 = [qp.expect(self.Q, i) for i in state2_in_rot_frame]


    def build_drive_Hamiltonian_SW(self):
        return [self.cavity_sw.dag(),self.drive_cavity_1],[self.cavity_sw,self.drive_cavity_2]

    def build_drive_Hamiltonian(self):
        return [self.cavity_sw.dag(), self.drive_cavity_1], [self.cavity_sw, self.drive_cavity_2]


    def run_Disspersive(self):
        print('run dispersive')
        drive = self.build_drive_Hamiltonian_SW()
        Hamiltonian_with_drive = [self.build_Hamiltonian_disp(), drive[0], drive[1]]

        data_psi1 = qp.mesolve(Hamiltonian_with_drive, self.psi1_sw, self.times, c_ops=[np.sqrt(0) * self.cavity_sw],e_ops=[])
        data_psi2 = qp.mesolve(Hamiltonian_with_drive, self.psi2_sw, self.times, c_ops=[np.sqrt(0) * self.cavity_sw],e_ops=[])
        print('done dispersive')

        self.states_Disp_1 = data_psi1.states
        self.states_Disp_2 = data_psi2.states

        print('Hallo')

    def run_JCM(self):
        drive = self.build_drive_Hamiltonian()
        Hamiltonian_with_drive = [self.build_JCM(), drive[0], drive[1]]

        data_psi1 = qp.mesolve(Hamiltonian_with_drive, self.psi1, self.times, c_ops=[np.sqrt(0) * self.cavity],e_ops=[])
        data_psi2 = qp.mesolve(Hamiltonian_with_drive, self.psi2, self.times, c_ops=[np.sqrt(0) * self.cavity],e_ops=[])

        self.states_JCM_1 = data_psi1.states
        self.states_JCM_2 = data_psi2.states


    def plot(self):
        plt.figure(1)
        plt.plot(self.I_values_disp_state1 , self.Q_values_disp_state1 , self.I_values_disp_state2, self.Q_values_disp_state2,label='Dispersive' )
        plt.plot(self.I_values_JCM_state1, self.Q_values_JCM_state1, self.I_values_JCM_state2,self.Q_values_JCM_state2, label='JCM')
        plt.title('Homodyne time evolution')
        plt.xlabel('I')
        plt.grid()
        plt.ylabel('Q')
        plt.legend()


        plt.show()




Atom_cavity= setup_Atom_Cavity(drive_frequenz=4*np.power(10,9)*2 * np.pi,
                                resonator_frequenz=4*np.power(10,9)*2 * np.pi,
                                qubit_frequenz=6.01*np.power(10,9)*2 * np.pi,
                                drive_ampltidue=10*np.power(10,6)*2 * np.pi,
                                interaction_strength=5*np.power(10,6)*2 * np.pi)


time = np.linspace(0.0, 0.0000000025, 100)
test = setup_Dispersive_readout(30,5,time,Atom_cavity)
test.run_Disspersive()
test.run_JCM()
test.calculating_expectation_values(System='Disp')
test.calculating_expectation_values(System='JCM')
test.plot()
print('hello')




