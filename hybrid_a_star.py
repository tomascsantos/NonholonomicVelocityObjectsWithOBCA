#Hybrid A*

#Open set O: contians neighboring nodes of nodes alreadt expanded during search
#Closed set C: all nodes which have been conclusively processed


#Heuristic
"""
h(C) = d(C,goal)
Require consistent heuristics: h(A) <= h(B) + d(A,B)
"""

# Choose path by ordered list
"""
S: Starting Node
G: Goal Node
C: Current Node
h(C): heuristic from C to goal
d(a,b): actual distance from node a to node b

F = d(S, C)+ h(C) <- choose smallest
"""

import numpy as np
from queue import PriorityQueue
import vtkplotter as vtk
import math
from scipy.integrate import odeint

ROBOT_LENGTH = .5

class Node:
    """
    Define node as (pos, g, f, parent, state)
    """
    def __init__(self, pos, g, f, parent, connect):
        self.pos = pos
        self.discrete = discretize(pos)
        self.g = g
        self.f = f
        self.parent = parent #Node
        self.connecting_path = connect

def round_val(val):
    return round(val*5)/5

def discretize(pos):
    """
    Takes in a pos and returns a discretized version, based off of grid increments
    """
    x, y, theta, phi = round_val(pos[0]), round_val(pos[1]), round_val(pos[2]), round_val(pos[3])
    return [x, y, theta, phi]


def construct_path(node):
    """
    Takes in final node and returns list of states
    """
    path = []
    while node.parent:
        print("item ",node.connecting_path)
        print("shape ",node.connecting_path.shape)
        path.append(node.connecting_path.reverse())
        node = node.parent
    path.append(node.connecting_path.reverse())
    path.reverse()
    return path

def heur(start_pos, goal_pos):
    """
    returns heuristic from current node to final node
    """
    return distance(start_pos, goal_pos) -1

def dynam(y, t, a1, a2):
    """
    Dynamics for bicycle model, used as input to odeint
    """
    x,y,theta,phi = y[0], y[1], y[2], y[3]
    omega = np.pi
    p_prime = a2 * np.cos(omega * t)
    t_prime = a1 / ROBOT_LENGTH * np.tan(phi)
    x_prime = a1 * np.cos(theta) 
    y_prime = a1 * np.sin(theta)
    return [x_prime, y_prime, t_prime, p_prime]

def distance(c1, c2):
    """
    returns distance between 2 pos
    c1 and c2 should be numpy.ndarrays of size (4,)
    """
    x1,x2 = c1[0], c2[0]
    y1,y2 = c1[1], c2[1]
    theta1,theta2 = c1[2], c2[2]
    p1, p2 = c1[:-1], c2[:-1]
    diff = abs(p1[2] - p2[2]) % (2*np.pi)
    eucl = math.sqrt((x1 - x2)**2 + (y1-y2)**2)
    angle = min(diff, 2*np.pi - diff)
    
    # tune if needed
    alpha = 1
    beta = 1
    return alpha * eucl + beta * angle

def next_pos(vp, curr_node, start_pos, goal_pos):
    """
    returns states that can be reached from current state in time step dt
    """
    total_time = 1
    dt = .01
    a1s = np.linspace(-0.5, 0.5, num=11) #vel constraints
    a2s = np.linspace(-0.8, 0.8, num=11) #omega constraints
    times = np.arange(0, total_time, dt)

    new_nodes = []

    for a1 in a1s:
        if a1 != 0:
            for a2 in a2s:
                sol = odeint(dynam, np.array(curr_node.pos), times, args=(a1,a2))
                last_pos = list(sol[-1])
                vp += [vtk.shapes.Circle(last_pos[:2]+[0], c="blue", alpha=1, r=.01)]
                vp.show(interactive=0)
                g = distance(start_pos, last_pos)
                new_node = Node(last_pos, g, g + heur(last_pos, goal_pos), curr_node, sol)
                new_nodes.append(new_node)

    print("EXIT")
    return new_nodes

def is_obstacle(grid, curr):
    return False


def updateNeighbors(vp, grid, constraints, OPEN, CLOSED, closed_disc, open_disc, OPEN_l, n, start_pos, goal_pos):
    """
    Helper for a*
    """
    next_nodes = next_pos(vp, n, start_pos, goal_pos)
    for node in next_nodes:
        print("Node -> ", node.discrete)
        if node.discrete not in closed_disc:
            print("ANOTHER ________________________")
            if is_obstacle(grid, node):

                CLOSED.append(node)
                closed_disc.append(node.discrete)

            elif node.discrete in open_disc:
                print("Entered Slow")
                # Very slow
                g_prime = node.g
                for old in OPEN_l:
                    if node.discrete == old.discrete and g_prime < old.g:
                        new_pq = PriorityQueue()
                        while not OPEN.empty():
                            item = OPEN.get()[1]
                            if item.discrete == old.discrete:
                                new_pq.put((node.f, node))
                            else:
                                new_pq.put((item.f, item))
                        OPEN = new_pq
                        open_disc.append(node.discrete)
                        OPEN_l.append(node)
                        break
            else:
                vp += [vtk.shapes.Circle(node.pos[:2]+[0], c="orange", alpha=1, r=.06)]
                OPEN.put((node.f, node))
                open_disc.append(node.discrete)
                OPEN_l.append(node)
        
    return OPEN, CLOSED, closed_disc, open_disc, OPEN_l

def hybrid_a_star(vp, grid, constraints, start_pos, goal_pos):
    """
    Arguments:
        map: map of current state including obstacles
        constraints: constraints of bicycle model with turning, phi
        start_pos: x,y,theta,phi
        goal_pos: x,y,theta,phi

    Returns:
        list of states x,y, theta: path from start_pos to goal_pos
    """

    node = Node(start_pos, 0, heur(start_pos, goal_pos), None, None)
    OPEN = PriorityQueue()
    OPEN.put((node.f, node))

    OPEN_l = []
    OPEN_l.append(node)

    open_disc = []
    open_disc.append(node.discrete)

    CLOSED = []
    closed_disc = []

    count = 0
    while not OPEN.empty():
        print("Run again ", count)
        count += 1
        curr = OPEN.get()[1] #Get node with smallest f
        CLOSED.append(curr)
        closed_disc.append(curr.discrete)

        if distance(curr.pos, goal_pos) <= 1: # <- within threshold
            return construct_path(curr)
        else:
            OPEN, CLOSED, closed_disc, open_disc, OPEN_l = updateNeighbors(vp, grid, constraints, OPEN, CLOSED, closed_disc, open_disc, OPEN_l, curr, start_pos, goal_pos)

    print("No path found!")
    return None


### TASKS ###

#1) given a node, check if it is in the closed set
#2) check if grid position is an obstacle (DO LAST)


#3) given a node's discrete position, check if this discrete position is in the open set
#4) replace item in set
#5) remove node with smallest f

if __name__ == "__main__":
    vp = vtk.Plotter(size=(1080, 720), axes=0, interactive=0)
    start = [10,10,0,0]
    goal = [15,15, 1, 0]
    vp += [vtk.shapes.Circle(start[:2]+[0], c="blue", alpha=1, r=.1)]
    vp += [vtk.shapes.Circle(goal[:2]+[0], c="blue", alpha=1, r=.1)]

    path = hybrid_a_star(vp, None, None, start, goal) #<- [[x,y,t,p], [x,...]...]
    for p in path:
        x = p[:2] + [0]
        vp += [vtk.shapes.Circle(x, c="blue", alpha=1, r=.08)]

    vp.show(interactive=1)










