
import vtkplotter as vtk
import numpy as np
# from: https://github.com/raphaelkba/Roboxi/blob/master/mpc.py
from mpc import NonlinearMPC
from collections import deque

#some inspiration from:
#https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py

DT = 0.05  # [s] time tick
WB = 2.5  # [m]
CAR_PADDING = 0.5
MAX_STEER = .52 #http://street.umn.edu/VehControl/javahelp/HTML/Definition_of_Vehicle_Heading_and_Steeing_Angle.htm
MAX_VEL = 50

HORIZON_SECS = 1

def rotate(theta):
    rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    return rot


class Agent():
    def __init__(self, map, name, state=[0,0,0,0,0]):
        self.map = map
        #state is x,y,theta,velocity,phi
        self.state = np.array(state)
        self.runge = np.copy(self.state)
        self.name=name
        self.radius = 2.5
        self.past_states = deque(maxlen=5)
        #initialize points for visualization
        pos = np.concatenate((state[0:2],[0]))
        assert(len(pos) == 3)

        """
        For creating the car bounding box, we assume the middle of the back
        axle is the origin. Then, make some points assuming that the car is at
        theta=0. Next, rotate everyting by theta degrees and add the position
        of the back axle to all the points
        """
        box = np.array([
            [0, CAR_PADDING, 0],
            [WB, CAR_PADDING, 0],
            [WB, -CAR_PADDING, 0],
            [0, -CAR_PADDING, 0],
        ])

        box = box @ rotate(state[2]).T
        box += pos
        shifted_box = np.concatenate(([box[-1]], box[:-1]))
        self.bounding_box = vtk.shapes.Lines(box, shifted_box, lw=3)
        map.vp += [self.bounding_box]

        vel = self.getVel3D()
        map.vp += [vtk.shapes.Arrow(pos, pos + vel, c="r")]

    def getPos3D(self):
        return np.array([self.state[0], self.state[1],0])

    def getVel3D(self):
        theta = self.state[2]
        return np.array(self.state[3] * np.array([np.cos(theta), np.sin(theta), 0]))

    def visVelocityObstacle(self):
        agents = self.map.get_neighbors(self.name)
        vp = self.map.vp
        A = []
        b = []
        for a in agents[:1]:
            a_pos = a.getPos3D()
            pos = self.getPos3D()

            #relative v arrow
            relVel = self.getVel3D() - a.getVel3D()
            vp += [vtk.shapes.Arrow(pos, pos+relVel, c="green")]
            perp = np.cross([0,0,1], pos-a_pos)
            perp = perp / np.linalg.norm(perp)
            perp *= a.radius + self.radius
            perp2 = -1*perp
            leg1 = a_pos - pos + perp
            leg2 = a_pos - pos + perp2
            leg1Normal = np.cross([0,0,1], leg1)
            leg2Normal = np.cross(leg2, [0,0,1])
            leg1Normal = leg1Normal / np.linalg.norm(leg1Normal)
            leg2Normal = leg2Normal / np.linalg.norm(leg2Normal)

            vp += [vtk.shapes.Arrow(pos, pos+leg1Normal, c="blue")]
            vp += [vtk.shapes.Arrow(pos, pos+leg2Normal, c="blue")]
            #note we are appending as row vectors here.
            A.append(leg1Normal)
            A.append(leg2Normal)
            #project the pos onto the normal for the offset.
            #the normals are defined with respect to relatie v,
            #so when we add a's v we're back in absolute v frame.
            b.append(leg1Normal @ (pos + a.getVel3D()))
            b.append(leg2Normal @ (pos + a.getVel3D()))

        #now we define the constraints like we would for the opt prob.
        A = np.array(A)
        b = np.array(b)

        X = np.array([[x,y] for x in np.linspace(-20, 20, num=50)
                            for y in np.linspace(-20,20, num=50)
        ])
        X = np.block([X, np.zeros((len(X),1))])
        #print("x shape: ", X.shape)
        #show = [print(x.shape) for x in X]
        if (len(A) != 0):
            X = [x for x in X if np.all(np.greater(A @ x - b, np.zeros(2)))]
        vp += [vtk.shapes.Sphere(pos=x, r=.1, alpha=1, c="pink") for x in X]

        return A, b


    #redefine using pfaffian constraints
    #and Euler Discretization of dynamics
    def dynamics_step(self, u):

        #subtract the translation from visualization points
        pts = np.array(self.bounding_box.points())
        pos = np.array([self.state[0], self.state[1], 0])
        pts = pts - pos

        #update the state of the car [x, y, theta, vel, phi]
        theta = self.state[2]
        phi = self.state[4]

        f = lambda x,u: np.array([
            x[3] * np.cos(x[2]),
            x[3] * np.sin(x[2]),
            x[3] * np.tan(x[4])/WB,
            u[0],
            u[1],
        ])

        state_dot = f(self.state, u)

        # k1 = f(self.state ,          u)
        # k2 = f(self.state + DT/2*k1, u)
        # k3 = f(self.state + DT/2*k2, u)
        # k4 = f(self.state + DT*k3,   u)
        # self.state = self.state + DT/6*(k1+2*k2+2*k3+k4)

        # g1 = np.array([np.cos(theta), np.sin(theta), 1/WB*np.tan(phi), 1, 0]) * u[0]
        # g2 = np.array([0,0,0,0,1]) * u[1]
        # state_dot = g1 + g2
        self.state = self.state + state_dot * DT

        # #check velocity
        # if abs(self.state[3]) >= MAX_VEL:
        #     self.state[3] = np.sign(self.state[3]) * MAX_VEL
        # #check steering angle
        # if abs(self.state[4]) >= MAX_STEER:
        #     self.state[4] = np.sign(self.state[4]) * MAX_STEER

        theta_f = self.state[2]
        d_theta = theta_f - theta

        #rotate visualization points around origin
        pts = pts @ rotate(d_theta).T

        #add back the new translation
        pos = np.array([self.state[0], self.state[1], 0])
        pts = pts + pos

        #add a trace of where the car has been
        self.map.vp += [vtk.shapes.Sphere(pos=pos, r=.1, alpha=.5, c="red")]

        #update the point visualizations
        self.bounding_box.points(pts)
        return self.state


class Map():

    agents = []

    def __init__(self, vp):
        self.vp = vp

    def add_agent(self, agent):
        self.agents += [agent]

    def get_neighbors(self, name):
        return [a for a in self.agents if a.name != name]

    def create_agent(self, name=None, state=[0,0,0,0,0]):
        if name == None:
            name = "agent_" + str(len(self.agents))
        a = Agent(self, name, state)
        self.add_agent(a)
        return a


def make_circle_path(num_points):
    path = np.array([[50*np.cos(np.pi * x - (np.pi / 2)),
                      50*np.sin(np.pi * x - (np.pi / 2))]
                      for x in np.linspace(0,1,num_points)]).T
    return path

def make_sinusoid_path(num_points):
    path = np.array([[  x,
                        -8 * (np.cos(x/8) - 1),]
                        for x in np.linspace(0,50,num_points)]).T
    return path


def plot_warm_start(warm_start, vp):
    warm_start_3d = np.block([
        [warm_start[:2,:]],
        [np.zeros(len(warm_start.T))]
    ])
    vp += [vtk.shapes.Tube(warm_start_3d.T, c="yellow", alpha=.3, r=.2)]

def plot_path(path, vp):
    z = np.zeros((1,len(path[0])))
    path = np.vstack((path, z))
    vp += [vtk.shapes.Circle(path[:,0]+[0], c="green", r=1)]
    vp += [vtk.shapes.Circle(path[:,-1]+[0], c="red", r=.5)]
    vp += [vtk.shapes.Tube(path.T, c="blue", alpha=1, r=.08)]


def make_line_path(num_points):
    #state is x,y,theta,velocity,phi
    path = np.linspace([-10,0], [10,0], num_points).T
    return path

def closest_path_point(path, state, vp):
    s = state[:2]
    #vp += [vtk.shapes.Circle(s+[0], c="purple", r=.3, alpha=1)]
    diff = path.T - s
    diff_norm = np.linalg.norm(diff, axis=1)
    i = np.argmin(diff_norm)
    #vp += [vtk.shapes.Circle(path[:,i]+[0], c="yellow", r=.3, alpha=1)]
    return path[:,i:]


import time
def follow_path(vp, map):
    #generate a wavy path and visualize
    num_points = 200
    #path = make_circle_path(num_points)
    #path = make_line_path(num_points)
    path = make_sinusoid_path(num_points)
    plot_path(path, vp)

    """Adding MPC from toolbox"""
    mpc = NonlinearMPC(HORIZON_SECS, 0.1, WB)

    #follow path with car
    a = map.create_agent("main", state=np.append(path[:,0],[0,0,0]))

    #while we're not at our destination yet
    norm = np.linalg.norm(a.state[:2] - path[:2, -1])
    while norm > 1:
        path = closest_path_point(path, a.state, vp)
        start_time = time.time()
        controls = mpc.MPC(a.state, path)
        time_f = time.time()
        print("optimization time: ", time_f - start_time)

        # for c in controls.T:
        print(controls[:,0])
        a.dynamics_step(controls[:,0])
        vp.show()

        norm = np.linalg.norm(a.state[:2] - path[:2, -1])

    print("GOAL REACHED")

    vp.show(interactive=1)

def go_around_box(vp, map):
    num_points = 200
    path = make_line_path(num_points)
    #path = make_sinusoid_path(num_points)
    plot_path(path, vp)

    """Define a box and visualize it"""
    #row vector normals
    A = np.array([
        [0,1],
        [0,-1],
        [1,0],
        [-1,0]
    ])

    b = np.matrix([2,0.5,.5,.5]).T
    #b = np.matrix([.5]).T
    print(b.shape)

    vels = np.random.normal(scale=1, size=(2,500))
    show = np.all(np.greater(b,A@vels), axis=0)
    vp += [vtk.shapes.Sphere(list(v)+[0], c="purple", r=.05)
            for v,s in zip(vels.T, show.T) if s]


    """Adding MPC from toolbox"""
    mpc = NonlinearMPC(HORIZON_SECS, 0.1, WB)
    a = map.create_agent("main", state=np.append(path[:,0],[0,0,0]))

    #while we're not at our destination yet
    norm = np.linalg.norm(a.state[:2] - path[:2, -1])
    while norm > 1:
        vp.show()
        path = closest_path_point(path, a.state, vp)
        start_time = time.time()
        controls = mpc.MPC(a.state, path, A ,b)
        time_f = time.time()
        print("optimization time: ", time_f - start_time)

        # for c in controls.T:
        print(controls[:,0])
        a.dynamics_step(controls[:,0])

        norm = np.linalg.norm(a.state[:2] - path[:2, -1])

    print("GOAL REACHED")

    vp.show(interactive=1)




def show_vel_obstacles(vp, map):
    a = map.create_agent("agent", state=np.array([0,0,np.pi/5,1,0]))
    o = map.create_agent("obstacle", state=np.array([0,8,-np.pi/8,1,0]))

    A, b = a.visVelocityObstacle()
    print(A.shape)
    print(b.shape)

    vels = np.random.normal(scale=5, size=(100,2))
    vels = np.block([vels, np.zeros((len(vels),1))])

    pos = a.getPos3D()
    [print(np.greater(A @ v - b, np.zeros(2))) for v in vels]
    vp += [vtk.shapes.Arrow(pos, pos+v, c="purple", s=.005)
            for v in vels if np.all(np.greater(A @ v - b, np.zeros(2)))]

    vp.show(interactive=1)
    for _ in range(100):
        time.sleep(DT)
        a.dynamics_step([1,0])
        o.dynamics_step([1,0])
        vp.show(interactive=0)

    vp.show(interactive=1)

def main():
    vp = vtk.Plotter(size=(1080, 720), axes=0, interactive=0)
    map = Map(vp)

    #show_vel_obstacles(vp, map)
    #follow_path(vp, map)
    go_around_box(vp, map)

    vp.show(interactive=1)







main()
