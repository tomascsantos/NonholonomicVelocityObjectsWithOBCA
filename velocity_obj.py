
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
MAX_STEER = .8 #http://street.umn.edu/VehControl/javahelp/HTML/Definition_of_Vehicle_Heading_and_Steeing_Angle.htm
MAX_VEL = 50

HORIZON_SECS = 2

def rotate(theta):
    rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    return rot

def rotate2D(theta):
    rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
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

        #draw the outline of the bounding box
        box = box @ rotate(state[2]).T
        box += pos
        shifted_box = np.concatenate(([box[-1]], box[:-1]))
        self.bounding_box = vtk.shapes.Lines(box, shifted_box, lw=3)
        map.vp += [self.bounding_box]

        #visualize the velocity
        vel = self.getVel3D()
        self.vel_arrow = vtk.shapes.Line(pos, pos + vel, c="r", lw=3)
        map.vp += [self.vel_arrow]

        """
        To create the convex set for obstacle collision, we define a set
        B = {y:Gy<=g} where G is the normals and g are the offsets. We start
        by creating 4 normals to define a rectangle. then we find the offsets
        by projecting onto the normals.
        """
        self.G = np.array([
            [-1,0],
            [0, -1],
            [1,0],
            [0,1],
        ])
        box_pts = np.array(self.bounding_box.points())
        self.G = self.G @ rotate2D(state[2]).T
        self.g = np.diagonal(self.G @ box_pts[:,:2].T).reshape(self.G.shape[0],1)


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
        planes = [] #for visualizing the constraint

        for a in agents[:1]:
            a_pos = a.getPos3D()
            pos = self.getPos3D()
            a_vel = a.getVel3D()

            #relative v arrow
            relVel = self.getVel3D() - a.getVel3D()
            #vp += [vtk.shapes.Arrow(pos, pos+relVel, c="green")]
            perp = np.cross([0,0,1], pos-a_pos)
            perp = perp / np.linalg.norm(perp)
            perp *= a.radius + self.radius
            perp2 = -1*perp
            # planes += [vtk.shapes.Arrow(a_pos, a_pos+perp, c="green")]
            # planes += [vtk.shapes.Arrow(a_pos, a_pos+perp2, c="green")]
            leg1 = a_pos - pos + perp
            leg2 = a_pos - pos + perp2
            leg1Normal = np.cross(leg1,[0,0,1])
            leg2Normal = np.cross([0,0,1], leg2)
            leg1Normal = leg1Normal / np.linalg.norm(leg1Normal)
            leg2Normal = leg2Normal / np.linalg.norm(leg2Normal)

            """
            create the truncation hyperplane location
            Formula: (((obstacle pos - agent pos) / T ) + agent pos) + a_vel

            TODO: must add the radius to both of them.
            """
            trunc_pt = (a_pos - pos) / HORIZON_SECS + pos + a_vel
            trunc_direction = (a_pos - pos) / np.linalg.norm(a_pos - pos)
            # radius_norm = (a.radius + self.radius) / HORIZON_SECS
            # trunc_dir_scaled = trunc_direction * radius_norm
            # trunc_pt -= trunc_dir_scaled
            trunc_direction *= -1 #want points on the other side of it.
            planes += [vtk.shapes.Circle(trunc_pt, r=.1, c="green")]
            planes += [vtk.shapes.Plane(trunc_pt, normal=trunc_direction, c="blue")]
            #note we are appending as row vectors here.
            A.append(leg1Normal[:2])
            A.append(leg2Normal[:2])
            # A.append(trunc_direction[:2])
            #project the pos onto the normal for the offset.
            #the normals are defined with respect to relatie v,
            #so when we add a's v we're back in absolute v frame.
            b.append(leg1Normal[:2] @ (pos[:2] + a_vel[:2]))
            b.append(leg2Normal[:2] @ (pos[:2] + a_vel[:2]))
            # b.append(trunc_direction[:2] @ (trunc_pt[:2]))

            planes += [vtk.shapes.Plane(pos=pos+a_vel, normal=leg1Normal, sx=3)]
            planes += [vtk.shapes.Plane(pos=pos+a_vel, normal=leg2Normal, sx=3)]
            # planes += [vtk.shapes.Plane(pos=trunc_pt+a_vel, normal=trunc_direction, sx=3)]

        #now we define the constraints like we would for the opt prob.
        A = np.array(A)
        b = np.array(b).reshape((len(b), 1))
        return A, b, planes


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
        self.G = self.G @ rotate2D(d_theta).T

        #add back the new translation
        pos = np.array([self.state[0], self.state[1], 0])
        pts = pts + pos
        self.g = np.diagonal(self.G @ pts[:,:2].T).reshape(self.G.shape[0],1)


        #add a trace of where the car has been
        self.map.vp += [vtk.shapes.Sphere(pos=pos, r=.1, alpha=.5, c="red")]

        #update the point visualizations
        self.bounding_box.points(pts)
        self.vel_arrow.points([pos, pos+self.getVel3D()])
        return self.state


    def visConvexBoundingBox(self):
        print("g", self.g)
        print("G", self.G)
        pos = np.array([self.state[0], self.state[1], 0])
        dots = np.random.normal(loc=pos[:2].reshape((2,1)), scale=1, size=(2,300))
        show = np.all(np.greater(self.g, self.G@dots), axis=0)
        points = [vtk.shapes.Circle(list(v)+[0], c="purple", r=.05)
                    for v,s in zip(dots.T, show.T) if s]
        return points


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
    vp += [vtk.shapes.Circle(path[:,0]+[0], c="green", r=.1)]
    vp += [vtk.shapes.Circle(path[:,-1]+[0], c="red", r=.5)]
    vp += [vtk.shapes.Tube(path.T, c="blue", alpha=1, r=.08)]


def make_line_path(num_points):
    #state is x,y,theta,velocity,phi
    path = np.linspace([-10,0], [10,0], num_points).T
    return path

def closest_path_point(path, state, vp):
    s = state[:2]
    diff = path.T - s
    diff_norm = np.linalg.norm(diff, axis=1)
    i = np.argmin(diff_norm)
    return path[:,i:]



import time
def follow_path(vp, map):
    #generate a wavy path and visualize
    num_points = 200
    #path = make_circle_path(num_points)
    #path = make_line_path(num_points)
    path = make_sinusoid_path(num_points)
    plot_path(path, vp)


    #follow path with car
    a = map.create_agent("main", state=np.append(path[:,0],[0,0,0]))

    vp.show(interactive=0)
    """Adding MPC from toolbox"""
    mpc = NonlinearMPC(HORIZON_SECS, 0.1, WB, vp)

    input("start")
    #while we're not at our destination yet
    norm = np.linalg.norm(a.state[:2] - path[:2, -1])
    while norm > 1:
        path = closest_path_point(path, a.state, vp)
        start_time = time.time()
        controls, viz = mpc.MPC(a.state, path, a)
        time_f = time.time()
        print("optimization time: ", time_f - start_time)

        # pts = a.visConvexBoundingBox()
        # vp += pts
        vp.show(interactive=0)
        # if len(pts) > 0:
        #     vp.clear(pts)

        a.dynamics_step(controls[:,0])
        norm = np.linalg.norm(a.state[:2] - path[:2, -1])

    print("GOAL REACHED")

    vp.show(interactive=1)

def go_around_moving_box(vp, map):
    """Generate the Path to follow"""
    num_points = 200
    path = make_line_path(num_points)
    #path = make_sinusoid_path(num_points)
    plot_path(path, vp)

    # """Define a box and visualize it"""
    # C = np.array([
    #     [0,1],
    #     [0,-1],
    #     [1,0],
    #     [-1,0]
    # ])
    # d = np.matrix([.5,0.5,.5,.5]).T
    # vels = np.random.normal(scale=1, size=(2,500))
    # show = np.all(np.greater(d,C@vels), axis=0)
    # vp += [vtk.shapes.Sphere(list(v)+[0], c="purple", r=.05)
    #         for v,s in zip(vels.T, show.T) if s]

    """Add the agents to the map and initialize controller"""
    a = map.create_agent("main", state=np.append(path[:,0],[0,0,0]))
    o_pos = a.state + [20, 0, -np.pi,2,0]
    o = map.create_agent("obstacle", state=o_pos)
    A, b, _ = a.visVelocityObstacle()
    mpc = NonlinearMPC(HORIZON_SECS, 0.1, WB, vp, A=A)
    vp.show()
    input("being visualization: [hit enter]")

    """Loop until arrive at destination"""
    norm = np.linalg.norm(a.state[:2] - path[:2, -1])
    times = []
    while norm > 1:
        A, b, planes = a.visVelocityObstacle()
        #visualize the plane and it's feasible region
        # vp += planes

        #truncate the path to the points in the future
        path = closest_path_point(path, a.state, vp)

        #time the optimization step
        start_time = time.time()
        controls, viz = mpc.MPC(a.state, path, a, A=A, b=b)
        time_f = time.time()
        vp += viz

        vp.show(interactive=0)
        # vp.clear(planes)
        if len(viz) > 0:
            vp.clear(viz)

        #update simulation
        a.dynamics_step(controls[:,0])
        o.dynamics_step([1,0])
        norm = np.linalg.norm(a.state[:2] - path[:2, -1])
        time_diff = time_f - start_time
        times.append(time_diff)

    print("GOAL REACHED")
    print("mean time: ", np.mean(times), "max: ", np.max(times), "median", np.median(times))
    input("Finished? [hit enter]")
    vp.show(interactive=1)


def go_around_box(vp, map):
    num_points = 200
    path = make_line_path(num_points)
    #path = make_sinusoid_path(num_points)
    plot_path(path, vp)

    """Define a box and visualize it"""
    #row vector normals
    C = np.array([
        [0,1],
        [0,-1],
        [1,0],
        [-1,0]
    ])
    d = np.matrix([.5,0.5,.5,.5]).T
    vels = np.random.normal(scale=1, size=(2,500))
    show = np.all(np.greater(d,C@vels), axis=0)
    vp += [vtk.shapes.Sphere(list(v)+[0], c="purple", r=.05)
            for v,s in zip(vels.T, show.T) if s]


    """Adding MPC from toolbox"""
    a = map.create_agent("main", state=np.append(path[:,0],[0,0,0]))
    mpc = NonlinearMPC(HORIZON_SECS, 0.1, WB, vp, C=C)

    vp.show()
    input("being visualization: [hit enter]")
    #while we're not at our destination yet
    norm = np.linalg.norm(a.state[:2] - path[:2, -1])
    while norm > 2:
        vp.show(interactive=0)
        path = closest_path_point(path, a.state, vp)
        start_time = time.time()
        controls, viz = mpc.MPC(a.state, path, a, C=C, d=d)
        time_f = time.time()
        print("optimization time: ", time_f - start_time)

        a.dynamics_step(controls[:,0])

        norm = np.linalg.norm(a.state[:2] - path[:2, -1])

    print("GOAL REACHED")

    vp.show(interactive=1)

def main():
    vp = vtk.Plotter(size=(1080, 720), axes=0, interactive=0)
    map = Map(vp)

    #follow_path(vp, map)
    #go_around_box(vp, map)
    go_around_moving_box(vp, map)

    vp.show(interactive=1)







main()
