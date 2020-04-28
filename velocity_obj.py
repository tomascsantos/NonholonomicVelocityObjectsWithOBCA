
import vtkplotter as vtk
import numpy as np
# from: https://github.com/raphaelkba/Roboxi/blob/master/mpc.py
from mpc import NonlinearMPC

#some inspiration from:
#https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/model_predictive_speed_and_steer_control.py

DT = 0.2  # [s] time tick
WB = 2.5  # [m]
CAR_PADDING = 0.5
MAX_STEER = .52 #http://street.umn.edu/VehControl/javahelp/HTML/Definition_of_Vehicle_Heading_and_Steeing_Angle.htm
MAX_VEL = 50


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
        self.name=name
        self.radius = .1
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
        map.vp += [vtk.shapes.Arrow(pos, pos + vel, c="b")]

    def getPos3D(self):
        return np.array([self.state[0], self.state[1],0])

    def getVel3D(self):
        theta = self.state[2]
        return np.array(self.state[3] * np.array([np.cos(theta), np.sin(theta), 0]))

    def visVelocityObstacle(self):
        agents = self.map.get_neighbors(self.name)
        vis = []
        A = []
        b = []
        for a in agents:
            a_pos = a.getPos3D()
            pos = self.getPos3D()

            #relative v arrow
            relVel = self.getVel3D() - a.getVel3D()
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

        X = np.random.normal(loc=pos[:2], scale=5, size=(5000,2))
        X = np.block([X, np.zeros((5000,1))])
        #show = [print(A @ x - b, x) for x in X]
        show = [x for x in X if np.all(np.greater(A @ x - b, np.zeros(2)))]
        vis += [vtk.shapes.Sphere(pos=s, r=.1, alpha=1, c="pink") for s in show]
        return vis


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

        state_dot = np.array([
            self.state[3] * np.cos(self.state[2]),
            self.state[3] * np.sin(self.state[2]),
            self.state[3] * np.tan(self.state[4])/WB,
            u[0],
            u[1],
        ])

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
        self.vp += vtk.Grid(sx=100, sy=100)

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


def make_circle_path(num_points, vp):
    path = np.array([[50*np.cos(np.pi * x + (np.pi / 2)),
                      50*np.sin(np.pi * x + (np.pi / 2)),0]
                      for x in np.linspace(0,1,num_points)])

    #create warm start traj
    start = path[0,:2]
    middle = path[len(path)//2,:2]
    end = path[-1,:2]
    print(start, ":", end)
    #transpose puts it in form that MPC wants: rows are states, columns are time.
    #the warm start is a warm start for the whole state: [x,y,theta,v,phi].T
    approx1 = np.linspace(start, middle, num=num_points//2)
    approx2 = np.linspace(middle, end, num=num_points//2)
    approx = np.concatenate((approx1, approx2))

    warm_start = np.block([
        [path[:,:2].T],
        [np.zeros((1,num_points))],
        [np.ones((1,num_points)) * 5],
        [np.zeros((1,num_points))]
    ])


    return warm_start, path

def plot_warm_start(warm_start, vp):
    warm_start_3d = np.block([
        [warm_start[:2,:]],
        [np.zeros(len(warm_start.T))]
    ])
    vp += [vtk.shapes.Tube(warm_start_3d.T, c="yellow", alpha=.3, r=.2)]

def plot_path(path, vp):
    vp += [vtk.shapes.Circle(path[0,:2]+[0], c="green", r=1)]
    vp += [vtk.shapes.Circle(path[-1,:2]+[0], c="red", r=.5)]


def make_line_path(num_points, vp):
    #state is x,y,theta,velocity,phi
    path = np.linspace([0,0,0], [20,20,0], num_points).T

    """Testing cost function"""
    """TODO: YUPPPP THE COST FUNCTION WAS ROTATED!!!! """
    print(len(path[0]))
    print(path[0][:].shape)

    warm_start = np.block([
        [path[:,:2].T],
        [np.ones((1,num_points)) * np.pi / 4], #45 degrees
        [np.ones((1,num_points)) * 2], #velocity of 2
        [np.zeros((1,num_points))] #steering angle of 0
    ])

    return warm_start, path


import time
def follow_path(vp, map):
    #generate a wavy path and visualize
    num_points = 200
    #warm_start, path = make_circle_path(num_points, vp)
    warm_start, path = make_line_path(num_points, vp)
    plot_warm_start(warm_start, vp)
    plot_path(path, vp)

    input("continue")
    """Adding MPC from toolbox"""
    mpc = NonlinearMPC(6, DT, WB)

    print(warm_start[:,0])
    #follow path with car
    a = map.create_agent("main", state=warm_start[:,0])
    controls = mpc.MPC(warm_start, path)
    print("controls: ", controls.shape)

    vp += [vtk.shapes.Tube(path, c="blue", r=.08)]

    for c in controls.T:
        print(c)
        print("state", a.state)
        time.sleep(DT)
        a.dynamics_step(c)
        vp.show()

    vp.show()


def show_vel_obstacles():
    a = map.create_agent("main", v=[5,2,0])
    o = map.create_agent("obstacle", pos=[0,5], v=[5,-1,0])

    vp += a.visVelocityObstacle()

    for _ in range(100):

        a.dynamics_step([1,0])
        vp.show()

def main():
    vp = vtk.Plotter(size=(1080, 720), axes=0, interactive=0)
    map = Map(vp)

    #show_vel_obstacles()
    follow_path(vp, map)

    vp.show(interactive=1)







main()
