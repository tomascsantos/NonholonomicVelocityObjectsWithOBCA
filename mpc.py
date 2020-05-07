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

DEFAULT_A = np.array([0,0]).reshape((1,2))

class NonlinearMPC():

    def __init__(self, N, dT, lr, vp, A=None, C=None):

        self.N = N # prediction horizon in seconds
        self.dT = dT # timestep
        self.H = int(N/dT) # prrdiction horizon steps
        print(self.H, "times steps")
        self.u_1 = np.zeros((self.H+1)) # acceleration control
        self.u_2 = np.zeros((self.H+1)) # steering velocity control
        self.warm_x = np.zeros((5,self.H+1))
        self.lr = lr
        self.vp = vp
        if (A is not None):
            self.warm_lam = np.zeros((A.shape[0], self.H+1))
        if (C is not None):
            self.warm_mu = np.zeros((C.shape[0], self.H+1))


    def MPC(self, states, path, A=None, b=None, C=None, d=None):
        """
        Inputs:
            A: polyhedron normals defining velocity object
            b: polyhedron offsets defining velocity object
            C: polyhedron normals defining static obstacles
            d: polyhedron offsets defining static obstacles
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
        #if we have velocity obstacles
        if (A is not None):
            lam = opti.variable(A.shape[0], self.H+1) #dual variables for obstacle opt
            slack = opti.variable(self.H+1) #dual variables for obstacle opt

        if (C is not None):
            mu = opti.variable(C.shape[0], self.H+1) #dual variables for obstacle opt
            slack2 = opti.variable(self.H+1)

        """Define Cost function"""
        def cost(i):
            total = 0
            #path tracking
            #distance from path
            total += 1 * ((x[i]-path[0][i])**2+(y[i]-path[1][i])**2)
            #distance from end
            total += .5 * ((x[i]-path[0][-1])**2+(y[i]-path[1][-1])**2)
            #shallow steering
            total += .1 *steer_angle[i]*steer_angle[i]
            #acceleration
            total += .01 * a[i] * a[i]
            #jerk
            if (i > 0):
                total += (a[i] - a[i-1])**2
            #velocity obstacles (slack)
            if (A is not None):
                total += 100 * slack[i]
            #static obstacles
            if (C is not None):
                total += 100 * slack2[i]
            return total

        V = 0
        for i in range(self.H+1):
            if i < len(path[0]):
                V += cost(i)
            else:
                V += cost(-1)
        opti.minimize(V)

        """System Dynamics"""
        f = lambda x,u: vertcat(x[3,:]*casadi.cos(x[2,:]),
                                x[3,:]*casadi.sin(x[2,:]),
                                x[3,:]*casadi.tan(x[4,:])/self.lr,
                                u[0,:],
                                u[1,:])

        """System Constraints"""
        opti.bounded(-math.pi, X[2,:], math.pi)
        opti.bounded(-50, X[3,:], 50)
        opti.bounded(-math.pi/3, X[4,:], math.pi/3)
        opti.subject_to(opti.bounded(-5.0, a, 5.0))
        opti.subject_to(opti.bounded(-math.radians(50.0), steer_angle, math.radians(50.0)))

        for k in range(self.H): # loop over control intervals
           k1 = f(X[:,k], U[:,k])
           x_next = X[:,k] + self.dT*k1
           opti.subject_to(X[:,k+1]==x_next)
           opti.subject_to(x[k] - x[k+1] < 1) # limit velocity


        """add the velocity obstacle constraint"""
        if (A is not None):
            for k in range(self.H+1): # loop over lambdas
                # (Ap - b)'lambda > 0
                vel_x = v[k] * (casadi.cos(theta[k])) + x[k]
                vel_y = v[k] * (casadi.sin(theta[k])) + y[k]
                Av = A[:,0] * vel_x + A[:,1] * vel_y

                opti.subject_to((Av-b).T @ lam[:,k] > -slack[k])
                opti.subject_to(lam[:,k] >= 0)
                opti.subject_to(slack[k] >= 0)
                norm = lam[:,k].T @ A @ A.T @ lam[:,k]
                opti.subject_to(norm == 1)

        """Add static obstacle constraint"""
        if (C is not None):
            for k in range(self.H+1): # loop over lambdas
                # (Ap - b)'lambda > 0
                opti.subject_to((C @ X[:2,k]-d).T @ mu[:,k] > - slack2[k])
                opti.subject_to(mu[:,k] >= 0)
                opti.subject_to(slack2[k] >= 0)
                #|A'lambda|_2 = 1
                norm = mu[:,k].T @ C @ C.T @ mu[:,k]
                opti.subject_to(norm == 1)

        """Initial Conditions"""
        opti.subject_to(x[0]==states[0])
        opti.subject_to(y[0]==states[1])
        opti.subject_to(theta[0]==states[2])
        opti.subject_to(v[0]==states[3])
        opti.subject_to(phi[0]==states[4])

        """Warm Start"""
        for n in range(self.H+1):
            opti.set_initial(U[0,n], self.u_1[n])
            opti.set_initial(U[1,n], self.u_2[n])
            opti.set_initial(X[:,n], self.warm_x[:,n])
            if (A is not None):
                opti.set_initial(lam[:,n], self.warm_lam[:,n])
            if (C is not None):
                opti.set_initial(mu[:,n], self.warm_mu[:,n])

        # solve NLP
        p_opts = {"expand":True}
        s_opts = {"max_iter": 1000,
                "hessian_approximation":"exact",
                "mumps_pivtol":1e-6,
                "alpha_for_y":"min",
                "recalc_y":"yes",
                "mumps_mem_percent":6000,
                "tol":1e-5,
                "print_level":0,
                "min_hessian_perturbation":1e-12,
                "jacobian_regularization_value":1e-7
        }
        opti.solver("ipopt", p_opts, s_opts)
        try:
            sol = opti.solve()
            print("acc: ",  sol.value(U[0,0]))
            print("steering: ",  sol.value(U[1,0]))
            control = np.array([sol.value(U[0,:]), sol.value(U[1,:])])

            """Populate Warm Start"""
            for i in range(self.H):
                self.u_1[i] = copy.deepcopy(sol.value(U[0,i+1]))
                self.u_2[i] = copy.deepcopy(sol.value(U[1,i+1]))
                self.warm_x[:,i] = copy.deepcopy(sol.value(X[:,i+1]))
                if (A is not None):
                    self.warm_lam[:,i] = copy.deepcopy(sol.value(lam[:,i+1]))
                if (C is not None):
                    self.warm_mu[:,i] = copy.deepcopy(sol.value(mu[:,i+1]))
            self.u_1[-1] = 0
            self.u_2[-1] = 0
            self.warm_x[:,-1] = np.zeros((5))

            """Begin Debugging Section"""
            x = sol.value(x)
            y = sol.value(X[1,:])
            theta = sol.value(X[2,:])
            v = sol.value(X[3,:])
            phi = sol.value(X[4,:])
            theta = sol.value(theta)
            if (A is not None):
                slack = sol.value(slack)
                print("Slack cost is: ", slack)
            if (C is not None):
                slack2 = sol.value(slack2)
                print("Slack cost is: ", slack2)

            """Visualizing Planned Velocities"""
            viz = []
            if (False and A is not None):
                # for i in range(len(v)):
                i=0
                pos = [x[i], y[i]]
                velxy = v[i] * np.array([np.cos(theta[i]), np.sin(theta[i])])
                velxy += pos
                viz += [vtk.shapes.Sphere(list(velxy)+[0], c="green", r=.1)]
                vels = np.random.normal(loc=velxy.reshape(2,1), scale=2.5, size=(2,200))
                show = np.all(np.greater(b, A@vels), axis=0)
                viz += [vtk.shapes.Sphere(list(v)+[0], c="purple", r=.05)
                        for v,s in zip(vels.T, show.T) if s]
                for i in range(len(v)):
                    pos = [x[i], y[i]]
                    velxy = v[i] * np.array([np.cos(theta[i]), np.sin(theta[i])])
                    velxy += pos
                    viz += [vtk.shapes.Sphere(list(velxy)+[0], c="green", r=.1)]

                    vel_x = v[i] * (casadi.cos(theta[i])) + x[i]
                    vel_y = v[i] * (casadi.sin(theta[i])) + y[i]
                    Av = A[:,0] * vel_x + A[:,1] * vel_y
                    # print("Av-b for ", i, " is: ", Av-b)
                    # print("Av-b.T @ lam is : ", (Av-b).T @ lam[:,i])
                    # print("-Slack is: ", -slack[i])
                # vels = np.random.normal(loc=pos.reshape((2,1)), scale=3, size=(2,300))
                # show = np.all(np.greater(b, A@vels), axis=0)
                # dots = [vtk.shapes.Sphere(list(v)+[0], c="purple", r=.05)
                #         for v,s in zip(vels.T, show.T) if s]
            return control, viz
        except:
            """Begin Debugging Section"""
            states = opti.debug.value(X)
            x = opti.debug.value(x)
            y = opti.debug.value(X[1,:])
            theta = opti.debug.value(X[2,:])
            v = opti.debug.value(X[3,:])
            phi = opti.debug.value(X[4,:])
            theta = opti.debug.value(theta)
            if (A is not None):
                slack = opti.debug.value(slack)
                print("Slack cost is: ", slack)
            if (C is not None):
                slack2 = opti.debug.value(slack2)
                print("Slack2 cost is: ", slack2)

            xys = states[:2,:].T
            self.vp += [vtk.shapes.Circle(pos=list(p)+[0],r=.1, c="darkred") for p in xys]
            print("NMPC failed", sys.exc_info())
            self.vp.show(interactive=1)
            input("finish")


        # in case it fails use previous computed controls and shift it
        control = np.array([self.u_1[0], self.u_2[0]])
        for i in range(self.H):
            self.u_1[i] = self.u_1[i+1]
            self.u_2[i] = self.u_2[i+1]
        self.u_1[-1] = 0
        self.u_2[-1] = 0
        return control
