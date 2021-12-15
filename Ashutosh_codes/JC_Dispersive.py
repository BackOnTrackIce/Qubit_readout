#%%
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
import math
import time

# %%

class system_setup:
    def __init__(self, qubit_freq, res_freq, drive_freq, drive_amp, g, kappa, mu1, sigma):
        self.wr = res_freq
        self.wa = qubit_freq
        self.g = g
        self.kappa = kappa
        self.wd = drive_freq
        self.drive_strength = drive_amp
        self.mu1 = mu1
        self.sigma = sigma
        self.Delta = self.wa - self.wr
        self.lamda = self.g/self.Delta
        self.chi = (self.g**2)/self.Delta
        self.chi_exact = self.chi * (1 - ((self.g/self.Delta)**2))
        self.xi = -(self.g**4)/(self.Delta**3)
        self.n_crit = 1/(4*(self.lamda**2))


class readout_setup:
    def __init__(self, Nr, Nq, cavity_photon_number, tlist, parameters, H_flag1, H_flag2, SW_flag1, SW_flag2, runs_flag):
        self.wr = parameters.wr
        self.wa = parameters.wa
        self.g = parameters.g
        self.kappa = parameters.kappa
        self.wd = parameters.wd
        self.drive_strength = parameters.drive_strength
        self.mu1 = parameters.mu1
        self.sigma = parameters.sigma
        self.Delta = parameters.Delta
        self.lamda = parameters.lamda
        self.chi = parameters.chi
        self.chi_exact = parameters.chi_exact
        self.xi = parameters.xi
        self.n_crit = parameters.n_crit
        self.H_flag1 = H_flag1
        self.H_flag2 = H_flag2
        self.SW_flag1 = SW_flag1
        self.SW_flag2 = SW_flag2
        self.Nr = Nr
        self.Nq = Nq
        self.cavity_photon_number = cavity_photon_number
        self.runs_flag = runs_flag

        self.a =  tensor(destroy(self.Nr),qeye(self.Nq))
        self.sz = tensor(qeye(self.Nr),sigmaz())
        self.sm = tensor(qeye(self.Nr),sigmam())
        self.sp = tensor(qeye(self.Nr),sigmap())
        self.I =  tensor(qeye(self.Nr),qeye(self.Nq))

        self.psi_plus = ( tensor(basis(self.Nr,self.cavity_photon_number),basis(self.Nq,0)) + tensor(basis(self.Nr,self.cavity_photon_number),basis(self.Nq,1)) )/np.sqrt(2) 
        self.psi_up = tensor(basis(self.Nr,self.cavity_photon_number),basis(Nq,1))
        self.psi_down = tensor(basis(self.Nr,self.cavity_photon_number),basis(Nq,0))

        self.tlist = tlist

        self.pi_e = basis(Nq,1) * basis(Nq,1).dag()
        self.Nq_oper = self.a.dag() * self.a + tensor(qeye(Nr),self.pi_e)

    def Dispersive_hamiltonian(self):
        return self.wr*self.a.dag()*self.a + (self.wa + 2*self.chi*(self.a.dag() * self.a + self.I/2)) * self.sz/2 + self.chi/2

    def JC_hamiltonian(self):
        return self.wr*self.a.dag()*self.a + self.wa * self.sz/2 + self.g * (self.a.dag() * self.sm + self.a * self.sp)

    def Exact_diagonal_hamiltonian(self):
        return self.wr * self.a.dag() * self.a + self.wa * self.sz/2 - (self.Delta/2) * (1 - (self.I + 4*(self.lamda**2)*self.Nq_oper).sqrtm())* self.sz

    def Exact_dispersive_hamiltonian(self):
        return (self.wr + self.xi) * self.a.dag() * self.a + (self.wa + 2*self.chi_exact*(self.a.dag()*self.a + self.I/2))*self.sz/2 + self.xi*((self.a.dag()*self.a)**2)* self.sz

    def Dispersive_hamiltonian_with_higher_order(self):
        return self.Dispersive_hamiltonian() + 0.5 * (self.lamda**2) * ( 2*( self.a * self.sp + self.a.dag() * self.sm )* self.sz - 4*(self.a.dag() * self.a)*(self.a * self.sp + self.a.dag() * self.sm) - 2*(self.a * self.sp + self.a.dag() * self.sm) )

    def drive_lab_frame(self,t,args):
        return 0.5 * self.drive_strength  * (math.tanh((t - self.mu1)/self.sigma) + 1) * np.exp(-1j*self.wd*t)

    def drive_lab_frame_conj(self,t,args):
        return 0.5 * self.drive_strength  * (math.tanh((t - self.mu1)/self.sigma) + 1) * np.exp(1j*self.wd*t)

    def solve_master_equation(self, H_flag, psi_SW_flag):
        if H_flag == "Dispersive":
            H0 = self.Dispersive_hamiltonian()
        elif H_flag == "JC":
            H0 = self.JC_hamiltonian()
        elif H_flag == "Exact Diagonal":
            H0 = self.Exact_diagonal_hamiltonian()
        elif H_flag == "Exact Dispersive":
            H0 = self.Exact_dispersive_hamiltonian()
        elif H_flag == "Dispersive with higher order":
            H0 = self.Dispersive_hamiltonian_with_higher_order()
        
        H1 = self.a.dag()
        H2 = self.a

        H_total = [H0,[H1,self.drive_lab_frame], [H2, self.drive_lab_frame_conj]]

        c_ops = [np.sqrt(self.kappa) * self.a]
        e_ops = []

        if psi_SW_flag == True:
            psi = self.SW_unitary() * self.psi0
        else:
            psi = self.psi0
        
        output  = mesolve(H_total, psi, self.tlist, c_ops, e_ops, progress_bar=True, options= Options(nsteps=3000))
        return output

    def SW_unitary(self):
        exponent = self.lamda * (self.a * self.sp - self.a.dag()*self.sm )
        U = exponent.expm()
        return U

    def schrieffer_wolf_transform(self,Operator):
        U = self.SW_unitary()
        return U * Operator * U.dag()  
    
    def calc_dynamics(self, output, a, sz):
        state = output.states
        dynamics_res = [expect(a.dag() * a,i) for i in state]
        dynamics_qubit = [expect(sz,i) for i in state]
        return dynamics_res, dynamics_qubit

    def plot_dynamics(self, output1, output2):
        if self.SW_flag1 == True:
            dynamics1_res, dynamics1_qubit = self.calc_dynamics(output1, self.a.transform(self.SW_unitary()), self.sz.transform(self.SW_unitary()))
        else:
            dynamics1_res, dynamics1_qubit = self.calc_dynamics(output1, self.a, self.sz)
        
        if self.SW_flag2 == True:
            dynamics2_res, dynamics2_qubit = self.calc_dynamics(output2, self.a.transform(self.SW_unitary()), self.sz.transform(self.SW_unitary()))
        else:
            dynamics2_res, dynamics2_qubit = self.calc_dynamics(output2, self.a, self.sz)

        plt.figure(dpi = 200)
        plt.plot(self.tlist,dynamics1_res  ,label = self.H_flag1 + "Resonator")
        plt.plot(self.tlist,dynamics2_res  ,label = self.H_flag2 + "Resonator")
        plt.plot(self.tlist,dynamics1_qubit,label = self.H_flag1 + "Qubit")
        plt.plot(self.tlist,dynamics2_qubit,label = self.H_flag2 + "Qubit")
        plt.legend()
        plt.show()

    def frame_of_drive(self, output, SW_flag):
        state = output.states
        
        if SW_flag == True:
            a1 = self.a.transform(self.SW_unitary())
            sz1 = self.sz.transform(self.SW_unitary())
        else:
            a1 = self.a
            sz1 = self.sz

        U_exp = [1j * t * self.wd * (a1.dag() * a1 + sz1/2 ) for t in self.tlist]

        U_matrix = [i.expm() for i in U_exp]

        state_rot = [U_matrix[i] * state[i] * U_matrix[i].dag()  for i in range(len(self.tlist))]
        
        return state_rot
    
    def partial_trace_over_qubit(self, state0, state1):
        state1_res = [i.ptrace(0) for i in state0]
        state2_res = [i.ptrace(0) for i in state1]
        return state1_res, state2_res

    def calc_I_Q(self, state1_res, state2_res):
        a_res = destroy(self.Nr)

        Q1 = [np.real(expect(a_res, i)) for i in state1_res]
        I1 = [np.imag(expect(a_res, i)) for i in state1_res]
        Q2 = [np.real(expect(a_res, i)) for i in state2_res]
        I2 = [np.imag(expect(a_res, i)) for i in state2_res]

        return Q1, I1, Q2, I2

    def plot_I_Q(self, fig, ax, Q1, I1, Q2, I2):

        ax.plot(Q1, I1, label = self.H_flag1)
        ax.plot(Q2, I2, label = self.H_flag2)
        ax.legend()


    def calculate_trace_distance(self,state1, state2):
        trace_distance = [tracedist(state1[i],state2[i]) for i in range(len(self.tlist))]
        return trace_distance
    
    def plotStatePopulation(self):
        H0 = self.JC_hamiltonian()
        
        H1 = self.a.dag()
        H2 = self.a

        H_total = [H0,[H1,self.drive_lab_frame], [H2, self.drive_lab_frame_conj]]
        e_ops = []

        self.psi0 = tensor(basis(self.Nr,self.cavity_photon_number),basis(self.Nq,0))

        self.output = sesolve(H_total, self.psi0, self.tlist, e_ops, progress_bar=True, options= Options(nsteps=3000))
        states = self.output.states
        pop = []
        N = len(np.array(states[0]))
        for i in range(len(self.tlist)):
            for j in range(N):
                try:
                    pop[i].append(np.abs(states[i][j])[0,0]**2)
                except:
                    pop.append([])
                    pop[i].append(np.abs(states[i][j])[0,0]**2)
        
        plt.figure(dpi=100)
        plt.plot(tlist, pop)
        plt.show()
        plt.close()

        dynamics_res, dynamics_qubit = self.calc_dynamics(self.output, self.a, self.sz)
        plt.plot(tlist, dynamics_res, label="Resonator")
        plt.plot(tlist, dynamics_qubit, label="Resonator")
        plt.legend()




    def run_simulation(self):
        if self.runs_flag == "Superposed":
            self.psi0 = self.psi_plus
            print("#################### Solving Master equation ####################")
            output1_u = self.solve_master_equation(self.H_flag1, self.SW_flag1)
            output2_u = self.solve_master_equation(self.H_flag2, self.SW_flag2)
            
            print("#################### Plotting dynamics ####################")
            self.plot_dynamics(output1_u, output2_u)
            

            print("#################### Calculating I Q  ####################")
            state1_u = self.frame_of_drive(output1_u, self.SW_flag1)
            state2_u = self.frame_of_drive(output2_u, self.SW_flag2)

            state1_res, state2_res = self.partial_trace_over_qubit(state1_u, state2_u)
            Q1_u, I1_u, Q2_u, I2_u = self.calc_I_Q(state1_res, state2_res)

            print("#################### Plotting I Q ####################")
            fig, ax = plt.subplots(1,1,dpi=200)
            self.plot_I_Q(fig, ax, Q1_u, I1_u, Q2_u, I2_u)
            fig.show()
            

            print("#################### calculating trace distance ####################")
            
            trace_dist = self.calculate_trace_distance(state1_u,state2_u)

            print("#################### plotting trace distance ####################")
            
            plt.figure(dpi=200)
            plt.plot(self.tlist, trace_dist)
            plt.show()

        elif self.runs_flag == "Seperate":
            self.psi0 = self.psi_up
            print("#################### Solving Master equation ####################")
            output1_u = self.solve_master_equation(self.H_flag1, self.SW_flag1)
            output2_u = self.solve_master_equation(self.H_flag2, self.SW_flag2)

            self.psi0 = self.psi_down
            print("#################### Solving Master equation ####################")
            output1_d = self.solve_master_equation(self.H_flag1, self.SW_flag1)
            output2_d = self.solve_master_equation(self.H_flag2, self.SW_flag2)
            
            print("#################### Plotting dynamics ####################")
            self.plot_dynamics(output1_u, output2_u)
            

            print("#################### Calculating I Q  ####################")
            state1_u = self.frame_of_drive(output1_u, self.SW_flag1)
            state2_u = self.frame_of_drive(output2_u, self.SW_flag2)
            
            state1_d = self.frame_of_drive(output1_d, self.SW_flag1)
            state2_d = self.frame_of_drive(output2_d, self.SW_flag2)

            state1_res, state2_res = self.partial_trace_over_qubit(state1_u, state2_u)
            Q1_u, I1_u, Q2_u, I2_u = self.calc_I_Q(state1_res, state2_res)

            state1_res_d, state2_res_d = self.partial_trace_over_qubit(state1_d, state2_d)
            Q1_d, I1_d, Q2_d, I2_d = self.calc_I_Q(state1_res_d, state2_res_d)

            print("#################### Plotting I Q ####################")
            fig, ax = plt.subplots(1,1,dpi=200)
            self.plot_I_Q(fig, ax, Q1_u, I1_u, Q2_u, I2_u)
            self.plot_I_Q(fig, ax, Q1_d, I1_d, Q2_d, I2_d)
            fig.show()
            

            print("#################### calculating trace distance ####################")
            
            trace_dist = self.calculate_trace_distance(state1_u,state2_u)

            print("#################### plotting trace distance ####################")
            
            plt.figure(dpi=200)
            plt.plot(self.tlist, trace_dist)
            plt.show()

        else:
            print("runs_flag has to be Superposed or Seperate")


    def IQWithoutFrameRotation(self):
        self.psi0 = self.psi_up
        output1_u = self.solve_master_equation(self.H_flag1, self.SW_flag1)
        output2_u = self.solve_master_equation(self.H_flag2, self.SW_flag2)

        self.psi0 = self.psi_down
        output1_d = self.solve_master_equation(self.H_flag1, self.SW_flag1)
        output2_d = self.solve_master_equation(self.H_flag2, self.SW_flag2)
        
        self.plot_dynamics(output1_u, output2_u)
        

        print("#################### Calculating I Q  ####################")
        state1_u = output1_u.states
        state2_u = output2_u.states
        
        state1_d = output1_d.states
        state2_d = output2_d.states

        state1_res, state2_res = self.partial_trace_over_qubit(state1_u, state2_u)
        Q1_u, I1_u, Q2_u, I2_u = self.calc_I_Q(state1_res, state2_res)

        state1_res_d, state2_res_d = self.partial_trace_over_qubit(state1_d, state2_d)
        Q1_d, I1_d, Q2_d, I2_d = self.calc_I_Q(state1_res_d, state2_res_d)

        print("#################### Plotting I Q ####################")
        fig, ax = plt.subplots(1,1,dpi=200)
        self.plot_I_Q(fig, ax, Q1_u, I1_u, Q2_u, I2_u)
        self.plot_I_Q(fig, ax, Q1_d, I1_d, Q2_d, I2_d)
        fig.show()
        

        print("#################### calculating trace distance ####################")
        
        trace_dist = self.calculate_trace_distance(state1_u,state2_u)

        print("#################### plotting trace distance ####################")
        
        plt.figure(dpi=200)
        plt.plot(self.tlist, trace_dist)
        plt.show()


        
        



#%%
tmax = 100 * 1e-9
Npoints = 1000
tlist = np.linspace(0,tmax,Npoints)
# %%

H_flag_list = ["Dispersive","JC","Exact Diagonal","Exact Dispersive","Dispersive with higher order"]

Test1 = system_setup(
                     qubit_freq = 2 * np.pi* 6 * 1e9,
                     res_freq = 2 * np.pi * 4 * 1e9,
                     drive_freq = 2 * np.pi * 4 * 1e9,
                     drive_amp = 2 * np.pi * 0.01 * 1e9,
                     g = 2 * np.pi * 50 * 1e6,
                     kappa= 2 * np.pi * 0.0025 * 1e9,
                     mu1 = 10 * 1e-6/(2 * np.pi),
                     sigma = 10 * 1e-6/(2 * np.pi)
)

Test1_readout = readout_setup(
                Nr = 20,
                Nq = 2,
                cavity_photon_number = 0,
                tlist = tlist,
                parameters = Test1,
                H_flag1 = H_flag_list[1],
                H_flag2 = H_flag_list[0],
                SW_flag1 = False,
                SW_flag2 = True,
                runs_flag = "Seperate"                
)
# %%
Test1_readout.IQWithoutFrameRotation()
#%%
Test1_readout.plotStatePopulation()

#%%
Test1_readout.run_simulation()

# %%
