# Sample code from http://www.redblobgames.com/pathfinding/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>

##TODO: find code for strings of arrows that point in diagonal directions

import collections
import numpy as np



#class SimpleGraph:
#    def __init__(self):
#        self.edges = {}
    
#    def neighbors(self, id):
#        return self.edges[id]

#example_graph = SimpleGraph()
#example_graph.edges = {
#    'A': ['B'],
#    'B': ['A', 'C', 'D'],
#    'C': ['A'],
#    'D': ['E', 'A'],
#    'E': ['B']
#}


### This Queue is just used for Breadth first search
### The Priority Queue class is used to define weights for each vertex
#class Queue:
#    def __init__(self):
#        self.elements = collections.deque()
#    
#    def empty(self):
#        return len(self.elements) == 0
    
#    def put(self, x):
#        self.elements.append(x)
    
#    def get(self):
#        return self.elements.popleft()



# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)

def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = "\u2192"
        if x2 == x1 - 1: r = "\u2190"
        if y2 == y1 + 1: r = "\u2193"
        if y2 == y1 - 1: r = "\u2191"
        #TODO: find the code for printing diagonal arrows pointing to parent nodes
        #if (x2 == x1+1 and y2 == y1): r = "\u2192" #it should look kind of like this
    if 'start' in style and id == style['start']: r = "S" #symbol displaying start node
    if 'goal' in style and id == style['goal']: r = "G"   #symbol displaying goal node
    if 'path' in style and id in style['path']: r = "o"   #symbol displaying path nodes
    if id in graph.walls: r = "#" * width
    return r

def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()

# data from main article
# DIAGRAM1_WALLS = [from_id_width(id, width=30) for id in [21,22,51,52,81,82,93,94,111,112,123,124,133,134,141,142,153,154,163,164,171,172,173,174,175,183,184,193,194,201,202,203,204,205,213,214,223,224,243,244,253,254,273,274,283,284,303,304,313,314,333,334,343,344,373,374,403,404,433,434]]

class SquareGrid:
    def __init__(self, width, height,p):
        self.width = width
        self.height = height
        self.p=p # smallest step size/resolution
        self.walls = []
    
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id):
        return id not in self.walls

    def nearest_node(self, id):
        (x,y)=id
        return round(x*(1/self.p))*self.p, round(y*(1/self.p))*self.p

    def low_vib_passable(self, id):
        (xn, yn) = self.nearest_node(id)
        (xc, yc) = self.nearest_node(self.current_node)
        x_points=abs((xn-xc)/self.p)+1
        y_points=abs((yn-yc)/self.p)+1
        if x_points >= y_points: # motion in horizontal direction
            x_along=np.linspace(xc,xn,x_points)
            y_along=np.linspace(yc,yn,x_points)
        else: # x_points < y_points   # motion in vertical direction
            x_along=np.linspace(xc,xn,y_points)
            y_along=np.linspace(yc,yn,y_points)

        tru_fal=[]
        for i,j in zip(x_along,y_along):
            tru_fal.append(not (i,j) in self.walls) # False if it intersects wall

        return not False in tru_fal

    def ob_close(self, current, goal):
        (xg, yg) = goal
        (xc, yc) = current

        xlow=min(xg,xc)
        ylow=min(yg,yc)

        xhigh=min(xg,xc)
        yhigh=min(yg,yc)


        between = []
        for i,j in self.walls:
            between.append(xlow < i < xhigh and ylow < j < yhigh) # If true, then wall node is between current and goal

        return True in between # If any True in between, then there is an obstacle in the way


    
    def neighbors(self, id): # make changes to allow different grid resolutions
        (x, y) = id
        #results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)] #only allows four directions of motion
        #Define diagonal nodes as neighbors
        # results = [(x+1, y), (x+1, y-1), (x, y-1), (x-1, y-1), (x-1, y), (x-1, y+1), (x, y+1), (x+1, y+1)] #allows 8 directions of motion
        results = [(x+self.p, y), (x+self.p, y-self.p), (x, y-self.p), (x-self.p, y-self.p), (x-self.p, y), (x-self.p, y+self.p), (x, y+self.p), (x+self.p, y+self.p)] #allows 8 directions of motion
        if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    def low_vib_neighbors(self, id, v, a, tau):
        (x, y) = id # current node
        self.current_node = id # So that it can be used later
        #Define nodes which are v*tau away from current node as neighbors
        straight=v*tau # motion in straight line
        diag=v*(np.sqrt(2)/2)*tau # diagonal motion
        # the neighbors are
        results = [(x+straight, y), (x+diag, y-diag), (x, y-straight), (x-diag, y-diag), (x-straight, y), (x-diag, y+diag), (x, y+straight), (x+diag, y+diag)]
        if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.low_vib_passable, results)
        return results

class GridWithWeights(SquareGrid):
    def __init__(self, width, height, p):
        super().__init__(width, height, p)
        self.weights = {}

    def cost(self, from_node, to_node):
        (x1,y1) = from_node
        (x2,y2) = to_node
        d=np.sqrt((x2-x1)**2 + (y2-y1)**2)
        return self.weights.get(to_node, d)


def vib_cost(v,start,next,current,came_from,time,param):

    m,k,zeta = param
    vib_cost_x=0.0
    vib_cost_y=0.0

    wn=np.sqrt(k/m)
    wd=wn*np.sqrt(1-zeta**2)


    # Do calculation of the first step before the while loop
    (xn, yn)=next
    (xc, yc)=current

    d=np.sqrt((xn-xc)**2+(yn-yc)**2) # distance between next and current nodes
    dt=d/v # time to get from there to here
    time[next]=time[current]+dt

    resp_x = (-1)*(xn-xc)*(np.exp(-zeta*wn*(time[next]-time[next]))*np.cos(wd*(time[next]-time[next]))-1)
    resp_y = (-1)*(yn-yc)*(np.exp(-zeta*wn*(time[next]-time[next]))*np.cos(wd*(time[next]-time[next]))-1)

    vib_cost_x=np.sum([vib_cost_x,resp_x])
    vib_cost_y=np.sum([vib_cost_y,resp_y])

    # start from current node
    here = current # here will keep going back
    while here!=start:
        there=came_from[here] # parent of here
        (x2,y2)=here
        (x1,y1)=there

        resp_x = (-1)*(x2-x1)*(np.exp(-zeta*wn*(time[next]-time[here]))*np.cos(wd*(time[next]-time[here]))-1)

        resp_y = (-1)*(y2-y1)*(np.exp(-zeta*wn*(time[next]-time[here]))*np.cos(wd*(time[next]-time[here]))-1)

        vib_cost_x=np.sum([vib_cost_x,resp_x])
        vib_cost_y=np.sum([vib_cost_y,resp_y]) # This is actually just the sum of the responses
        here=came_from[here] # parent node becomes here

    vib_cost=np.sqrt((vib_cost_x-xn)**2 + (vib_cost_y-yn)**2)


    return vib_cost

'''
def vib_cost(previous_node,current_node,next_node):
    (x1,y1) = previous_node
    (x2,y2) = current_node
    (x3,y3) = next_node
    if (x2-x1) == (x3-x2) and (y2-y1) == (y3-y2): # No change in direction
        vib_cost=0
    else:                                         # Direction of travel changes
        vib_cost=0.5 # Using values closer to 1 seems to create sub-optimal results
    return vib_cost
'''

'''
diagram4 = GridWithWeights(10, 10)
diagram4.walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
diagram4.weights = {loc: 5 for loc in [(3, 4), (3, 5), (4, 1), (4, 2),
                                       (4, 3), (4, 4), (4, 5), (4, 6), 
                                       (4, 7), (4, 8), (5, 1), (5, 2),
                                       (5, 3), (5, 4), (5, 5), (5, 6), 
                                       (5, 7), (5, 8), (6, 2), (6, 3), 
                                       (6, 4), (6, 5), (6, 6), (6, 7), 
                                       (7, 3), (7, 4), (7, 5)]}
'''

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    #path.append(start) # optional
    path.reverse() # optional
    return path

def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)  # This is Manhattan Distance
# TODO: Change Manhattan Distance to Euclidian distance

def a_star_search(graph, start, goal, v, p, param):
    frontier = PriorityQueue()
    frontier.put(start, 0) # put the start in the frontier
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    previous = start
    time={}
    time[start]=0.0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + 1*graph.cost(current, next) + 0*vib_cost(v,start,next,current,came_from,time, param) # vib_cost is used to add cost to changes in direction (I dont know if this will work)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next) # The heuristic is not included in the cost at each node and should not have a weighting factor
                frontier.put(next, priority)
                came_from[next] = current # When the next iteration comes around, came_from[next] (or came_from[next-1]) is previous
                previous = current

    
    return came_from, cost_so_far
