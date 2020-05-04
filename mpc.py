"""
Created on Fri Mar 15 01:10:46 2019
Nonlinear Model Predictive Control
#     CasADi -- A symbolic framework for dynamic optimization.
#     Copyright (C) 2010-2014 Joel Andersson, Joris Gillis, Moritz Diehl,
#                             K.U. Leuven. All rights reserved.
#     Copyright (C) 2011-2014 Greg Horn
#     CasADi is free software; you can redistribute it and/or
#     modify it under the terms of the GNU Lesser General Public
#     License as published by the Free Software Foundation; either
#     version 3 of the License, or (at your option) any later version.
#
#     CasADi is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#     Lesser General Public License for more details.
#
#     You should have received a copy of the GNU Lesser General Public
#     License along with CasADi; if not, write to the Free Software
#     Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
source: https://web.casadi.org/
@author: rapha
"""

import numpy as np
from sys import path
import vtkplotter as vtk
#path.append(r"C:\Users\rapha\Documents\Projects\casadi-windows-py36-v3.4.5-64bit")
from casadi import *
import copy
import math
import matplotlib.pyplot as plt

class NonlinearMPC():

    def __init__(self, N, dT, lr, A, vp):
        self.N = N # prediction horizon in seconds
        self.dT = dT # timestep
        self.H = int(N/dT) # prrdiction horizon steps
        print(self.H, "times steps")
        self.u_1 = np.zeros((self.H+1)) # acceleration control
        self.u_2 = np.zeros((self.H+1)) # steering velocity control
        self.warm_x = np.zeros((5,self.H+1))
        self.warm_lam = np.zeros((A.shape[0], self.H+1))
        self.lr = lr
        self.vp = vp

    def MPC(self, states, path, A, b):
        """
        Inputs:
            A: 2xk ndarray of the obstacle normal vectors
            b: 1xk ndarray of the obstacle offsets
        """

        opti = Opti() # Optimization problem

        # system states and controls
        X = opti.variable(5, self.H+1) # state trajectory
        x = X[0,:]
        y = X[1,:]
        theta = X[2,:]
        v = X[3,:]
        phi = X[4,:]

        U = opti.variable(2,self.H+1)   # control trajectory (acceleration and steering velocity)
        a = U[0,:]
        steer_angle = U[1,:]

        lam = opti.variable(A.shape[0], self.H+1) #dual variables for obstacle opt
        slack = opti.variable(self.H+1) #dual variables for obstacle opt




        """
        Notes: when I change the cost to just be the final point, instead of
        every point along the line it tried to avoid the obstacle but much
        to late.
        """
        def cost(i):
            distance_from_path = 1 * ((x[i]-path[0][i])**2+(y[i]-path[1][i])**2)
            distance_from_end = 8 * ((x[i]-path[0][-1])**2+(y[i]-path[1][-1])**2)
            shallow_steering = 1 *steer_angle[i]*steer_angle[i]
            speed = a[i] * a[i] / 2
            slack_cost = 100 * slack[i]
            #obst = .000000000001 / (A @ X[:2,i]-b).T @ lam[:,i]
            jerk, backwards = 0,0
            if (i > 0):
                jerk = (a[i] - a[i-1])**2

                        # if (v[i] < 0):
            #     backwards = -1*v[i]
            #backwards_motion = casadi.fmax(0, v[i] * -1)
            return speed + distance_from_path \
                    + shallow_steering + backwards + \
                    jerk + distance_from_end + slack_cost
        # cost function
        V = 0
        for i in range(self.H+1):
            if i < len(path[0]):
                """
                add much bigger weight for last couple points
                at the last value in H, the weight is 10x as important as first

                """
                V += cost(i)
            else:
                V += cost(-1)
    #    V += (casadi.fabs(casadi.cos(theta[-1]) - casadi.cos(pi))**2 + casadi.fabs(casadi.sin(theta[-1]) - casadi.sin(pi))**2)
        opti.minimize(V)

        # system
        f = lambda x,u: vertcat(x[3,:]*casadi.cos(x[2,:]),
                                x[3,:]*casadi.sin(x[2,:]),
                                x[3,:]*casadi.tan(x[4,:])/self.lr,
                                u[0,:],
                                u[1,:])

        # system constraints
        opti.bounded(-math.pi, X[2,:], math.pi)
        opti.bounded(-50, X[3,:], 50)
        opti.bounded(-math.pi/3, X[4,:], math.pi/3)
        opti.subject_to(opti.bounded(-5.0, a, 5.0))  # finish line at position 1
        opti.subject_to(opti.bounded(-math.radians(50.0), steer_angle, math.radians(50.0)))

        for k in range(self.H): # loop over control intervals
           # Runge-Kutta 4 integration
           k1 = f(X[:,k], U[:,k])
           x_next = X[:,k] + self.dT*k1
           opti.subject_to(X[:,k+1]==x_next) # close the gaps

        # for k in range(self.H): # loop over control intervals
        #    # Runge-Kutta 4 integration
        #    k1 = f(X[:,k],         U[:,k])
        #    k2 = f(X[:,k]+self.dT/2*k1, U[:,k])
        #    k3 = f(X[:,k]+self.dT/2*k2, U[:,k])
        #    k4 = f(X[:,k]+self.dT*k3,   U[:,k])
        #    x_next = X[:,k] + self.dT/6*(k1+2*k2+2*k3+k4)
        #    opti.subject_to(X[:,k+1]==x_next) # close the gaps


        """add the obstacle constraint OBCA"""
        for k in range(self.H): # loop over lambdas
            # (Ap - b)'lambda > 0
            opti.subject_to((A @ X[:2,k]-b).T @ lam[:,k] > -slack[k])
            opti.subject_to(lam[:,k] >= 0)
            opti.subject_to(slack[k] >= 0)
            #|A'lambda|_2 <= 1
            # tmp = A.T @ lam[:,k]
            # norm = tmp.T @ tmp
            norm = lam[:,k].T @ A @ A.T @ lam[:,k]
            opti.subject_to(norm <= 1)

        # initial conditions
        # opti.subject_to(x[0]==states[0,0])
        # opti.subject_to(y[0]==states[1,0])
        # opti.subject_to(theta[0]==states[2,0])
        # opti.subject_to(v[0]==states[3,0])
        # opti.subject_to(phi[0]==states[4,0])

        """do states as just an array"""
        opti.subject_to(x[0]==states[0])
        opti.subject_to(y[0]==states[1])
        opti.subject_to(theta[0]==states[2])
        opti.subject_to(v[0]==states[3])
        opti.subject_to(phi[0]==states[4])


        #initial x guesses.


        # initial control conditions
        for n in range(self.H+1):
            opti.set_initial(U[0,n], self.u_1[n])
            opti.set_initial(U[1,n], self.u_2[n])
            opti.set_initial(X[:,n], self.warm_x[:,n])
            opti.set_initial(lam[:,n], self.warm_lam[:,n])
            opti.set_initial(slack[:], np.zeros(self.H+1))

        # solve NLP
        p_opts = {"expand":True}
        s_opts = {"max_iter": 5000,
                "hessian_approximation":"exact",
                "mumps_pivtol":1e-6,
                "alpha_for_y":"min",
                "recalc_y":"yes",
                "mumps_mem_percent":6000,
                "max_iter":200,
                "tol":1e-5,
                "print_level":1,
                "min_hessian_perturbation":1e-12,
                "jacobian_regularization_value":1e-7
        }

        opti.solver("ipopt", p_opts, s_opts)
        try:
            sol = opti.solve()
            print("acc: ",  sol.value(U[0,0]))
            print("steering: ",  sol.value(U[1,0]))
            control = np.array([sol.value(U[0,:]), sol.value(U[1,:])])

            # shift controls
            for i in range(self.H):
                self.u_1[i] = copy.deepcopy(sol.value(U[0,i+1]))
                self.u_2[i] = copy.deepcopy(sol.value(U[1,i+1]))
                self.warm_x[:,i] = copy.deepcopy(sol.value(X[:,i+1]))
                self.warm_lam[:,i] = copy.deepcopy(sol.value(lam[:,i+1]))
            self.u_1[-1] = 0
            self.u_2[-1] = 0
            self.warm_x[:,-1] = np.zeros((5))
            #the line below this totally breaks things.
            #self.warm_lam[:,-1] = np.zeros((5))

            # ploting
            # from pylab import plot, step, figure, legend, show, spy
            # figure(1)
            # plot(sol.value(x[0:2]),sol.value(y[0:2]),label="speed")
            # plot(sol.value(x),sol.value(y),".")

            # figure(1)
            # plot(sol.value(lam),sol.value(y[0:2]),label="speed")
            # plot(sol.value(x),sol.value(y),".")
            # lam_val = sol.value(lam)
            x_val = sol.value(X)
            # print(lam_val)
            # for k in lam_val.shape[1]:
            #     print(k)
            #     const1 = (A @ x_val[:2,k]-b).T @ lam_val[:,k]
            #     tmp = A.T @ lam_val[:,k]
            #     norm = tmp.T @ tmp
            #     print(norm, "should be less than 1")
            #     print(const1 , "should be > 0")

            #figure()
            #spy(sol.value(jacobian(opti.g,opti.x)))
            #figure()
            #spy(sol.value(hessian(opti.f+dot(opti.lam_g,opti.g),opti.x)[0]))
            # del(opti)
            # show()
            # plt.pause(0.05)
            return control

        except:
            states = opti.debug.value(X)
            print("NMPC failed", sys.exc_info())
            xys = states[:2,:].T
            self.vp += [vtk.shapes.Circle(pos=list(p)+[0],r=.1, c="darkred") for p in xys]
            self.vp.show(interactive=1)


        # in case it fails use previous computed controls and shift it
        control = np.array([self.u_1[0], self.u_2[0]])
        for i in range(self.H):
            self.u_1[i] = self.u_1[i+1]
            self.u_2[i] = self.u_2[i+1]
        self.u_1[-1] = 0
        self.u_2[-1] = 0
        return control
