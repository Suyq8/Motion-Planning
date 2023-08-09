import numpy as np
import math
from pqdict import pqdict


def robotplanner(envmap, robotpos, targetpos):
    # all possible directions of the robot
    numofdirs = 8
    dX = [-1, -1, -1, 0, 0, 1, 1, 1]
    dY = [-1,  0,  1, -1, 1, -1, 0, 1]

    # use the old position if we fail to find an acceptable move
    newrobotpos = np.copy(robotpos)

    # for now greedily move towards the target
    # but this is the gateway function for your planner
    mindisttotarget = 1000000
    for dd in range(numofdirs):
        newx = robotpos[0] + dX[dd]
        newy = robotpos[1] + dY[dd]

        if (newx >= 0 and newx < envmap.shape[0] and newy >= 0 and newy < envmap.shape[1]):
            if(envmap[newx, newy] == 0):
                disttotarget = math.sqrt(
                    (newx-targetpos[0])**2 + (newy-targetpos[1])**2)
                if(disttotarget < mindisttotarget):
                    mindisttotarget = disttotarget
                    newrobotpos[0] = newx
                    newrobotpos[1] = newy
    return newrobotpos

class RobotPlanner():
    def __init__(self, env, N, K, epsilon):
        self.env = env  # environment
        self.N = N  # number of expansions
        self.H = np.full_like(env, -1)
        self.numofdirs = 8
        self.dX = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.dY = [-1, 0, 1, -1, 1, -1, 0, 1]
        self.dir = np.array([self.dX, self.dY]).T
        self.path = []
        self.count = 0
        self.K = K
        self.epsilon = epsilon

    def initialization(self, robotpos):
        self.close = set()
        self.open = pqdict()
        self.G = np.full_like(self.env, np.inf)
        self.G[robotpos[0], robotpos[1]] = 0

    def heuristic(self, currpos, targetpos):
        return np.linalg.norm(currpos-targetpos)
        # return np.sum(abs(currpos-targetpos))

    def is_valid(self, corr):
        if corr[0] >= 0 and corr[0] < self.env.shape[0] and corr[1] >= 0 and corr[1] < self.env.shape[1] and self.env[corr[0], corr[1]] == 0:
            return True
        else:
            return False

    def get_heuristic(self, currpos, targetpos):
        if self.H[currpos[0], currpos[1]] != -1:
            return self.H[currpos[0], currpos[1]]
        else:
            self.H[currpos[0], currpos[1]] = self.heuristic(currpos, targetpos)
            return self.H[currpos[0], currpos[1]]

    # RTAA* alogorithm
    def get_next_pos(self, robotpos, targetpos):
        if 0 < self.count < self.K and self.count < len(self.path):
            self.count += 1
            return self.path[-self.count+1]
        self.initialization(robotpos)
        parent = {}
        f = self.G[robotpos[0], robotpos[1]] + \
            self.epsilon*self.heuristic(robotpos, targetpos)
        robotpos = tuple(robotpos)
        self.open[robotpos] = f

        # expand
        for _ in range(self.N):
            if self.open.top()!=tuple(targetpos):
                corr, f = self.open.popitem()
            else:
                break

            self.close.add(corr)
            corr = np.array(corr)

            g = self.G[corr[0], corr[1]]+1
            for j in range(self.numofdirs):
                child = tuple(corr+self.dir[j])
                if self.is_valid(child) and child not in self.close:
                    if g < self.G[child[0], child[1]]:
                        self.G[child[0], child[1]] = g
                        h = self.get_heuristic(child, targetpos)
                        f = g+self.epsilon*h
                        if child in self.open:
                            if f < self.open[child]:
                                self.open.updateitem(child, f)
                        else:
                            self.open[child] = f
                        parent[child] = tuple(corr)

        # update heuristic
        goal, f_min = self.open.popitem()
        for x, y in self.close:
            self.H[x, y] = f_min - self.G[x, y]

        # get next position
        if self.K==1:
            curr = goal
            next = None
            while curr != robotpos:
                next = curr
                curr = parent[curr]
        else:
            self.path = []
            curr = goal
            next = None
            while curr != robotpos:
                self.path.append(curr)
                next = curr
                curr = parent[curr]
            self.count = 1

        return np.array(next)
