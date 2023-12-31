from targetplanner import targetplanner
from robotplanner import RobotPlanner
import time
import numpy as np
import math
from numpy import loadtxt
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.ion()


# functions to time how long planning takes

def tic():
    return time.time()


def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm, (time.time() - tstart)))


def runtest(mapfile, robotstart, targetstart, N, epsilon=1, K=1):
    # current positions of the target and robot
    robotpos = np.copy(robotstart)
    targetpos = np.copy(targetstart)

    # environment
    envmap = loadtxt(mapfile)

    # draw the environment
    # transpose because imshow places the first dimension on the y-axis
    '''
    f, ax = plt.subplots()
    ax.imshow(envmap.T, interpolation="none", cmap='gray_r', origin='lower',
              extent=(-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5))
    ax.axis([-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    hr = ax.plot(robotpos[0], robotpos[1], 'bs')
    ht = ax.plot(targetpos[0], targetpos[1], 'rs')
    f.canvas.flush_events()
    plt.show()
    '''

    # now comes the main loop
    numofmoves = 0
    caught = False
    planner = RobotPlanner(envmap, N, K, epsilon)
    trace_robot = robotpos
    trace_target = targetpos
    total_time = 0
    for i in tqdm(range(100000)):
        # call robot planner
        t0 = tic()
        newrobotpos = planner.get_next_pos(robotpos, targetpos)
        # compute move time for the target, if it is greater than 2 sec, the target will move multiple steps
        movetime = max(1, math.ceil((tic()-t0)/2.0))
        total_time += (tic()-t0)

        # check that the new commanded position is valid
        if (newrobotpos[0] < 0 or newrobotpos[0] >= envmap.shape[0] or
                newrobotpos[1] < 0 or newrobotpos[1] >= envmap.shape[1]):
            print('ERROR: out-of-map robot position commanded\n')
            break
        elif (envmap[newrobotpos[0], newrobotpos[1]] != 0):
            print('ERROR: invalid robot position commanded\n')
            break
        elif (abs(newrobotpos[0]-robotpos[0]) > 1 or abs(newrobotpos[1]-robotpos[1]) > 1):
            print('ERROR: invalid robot move commanded\n')
            break

        # call target planner to see how the target moves within the robot planning time
        newtargetpos = targetplanner(
            envmap, robotpos, targetpos, targetstart, movetime)

        # make the moves
        robotpos = newrobotpos
        targetpos = newtargetpos
        numofmoves += 1
        trace_robot = np.vstack([trace_robot, robotpos])
        trace_target = np.vstack([trace_target, targetpos])
        #print(robotpos, targetpos)
        #print(planner.H)

        # draw positions
        '''
        hr[0].set_xdata(robotpos[0])
        hr[0].set_ydata(robotpos[1])
        ht[0].set_xdata(targetpos[0])
        ht[0].set_ydata(targetpos[1])
        f.canvas.flush_events()
        plt.show()
        '''

        # check if target is caught
        if (abs(robotpos[0]-targetpos[0]) <= 1 and abs(robotpos[1]-targetpos[1]) <= 1):
            print('robotpos = (%d,%d)' % (robotpos[0], robotpos[1]))
            print('targetpos = (%d,%d)' % (targetpos[0], targetpos[1]))
            print(f'total time = {total_time}s')
            print(f'average moving time/step = {total_time/numofmoves}s')
            f, ax = plt.subplots()
            ax.imshow(envmap.T, interpolation="none", cmap='gray_r', origin='lower',
              extent=(-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5))
            ax.axis([-0.5, envmap.shape[0]-0.5, -0.5, envmap.shape[1]-0.5])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.plot(trace_robot[0, 0], trace_robot[0, 1], 'bs')
            ax.plot(trace_target[0, 0], trace_target[0, 1], 'rs')
            ax.plot(trace_robot[:, 0], trace_robot[:, 1], 'b')
            ax.plot(trace_target[:, 0], trace_target[:, 1], 'r')
            ax.legend(['robot', 'target'])
            ax.set_title(f'{mapfile[:-4]} (N={N}, epsilon={epsilon}, K={K})')
            plt.show()
            caught = True
            break

    return caught, numofmoves


def test_map0(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 2])
    targetstart = np.array([5, 3])
    return runtest('maps/map0.txt', robotstart, targetstart, N, epsilon, K)


def test_map1(N=4000, epsilon=1, K=1):
    robotstart = np.array([699, 799])
    targetstart = np.array([699, 1699])
    return runtest('maps/map1.txt', robotstart, targetstart, N, epsilon, K)


def test_map2(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 2])
    targetstart = np.array([7, 9])
    return runtest('maps/map2.txt', robotstart, targetstart, N, epsilon, K)


def test_map3(N=4000, epsilon=1, K=1):
    robotstart = np.array([249, 249])
    targetstart = np.array([399, 399])
    return runtest('maps/map3.txt', robotstart, targetstart, N, epsilon, K)


def test_map4(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 0])
    targetstart = np.array([5, 6])
    return runtest('maps/map4.txt', robotstart, targetstart, N, epsilon, K)


def test_map5(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 0])
    targetstart = np.array([29, 59])
    return runtest('maps/map5.txt', robotstart, targetstart, N, epsilon, K)


def test_map6(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 0])
    targetstart = np.array([29, 36])
    return runtest('maps/map6.txt', robotstart, targetstart, N, epsilon, K)


def test_map7(N=4000, epsilon=1, K=1):
    robotstart = np.array([0, 0])
    targetstart = np.array([4998, 4998])
    return runtest('maps/map7.txt', robotstart, targetstart, N, epsilon, K)


def test_map1b(N=4000, epsilon=1, K=1):
    robotstart = np.array([249, 1199])
    targetstart = np.array([1649, 1899])
    return runtest('maps/map1.txt', robotstart, targetstart, N, epsilon, K)


def test_map3b(N=4000, epsilon=1, K=1):
    robotstart = np.array([74, 249])
    targetstart = np.array([399, 399])
    return runtest('maps/map3.txt', robotstart, targetstart, N, epsilon, K)


def test_map3c(N=4000, epsilon=1, K=1):
    robotstart = np.array([4, 399])
    targetstart = np.array([399, 399])
    return runtest('maps/map3.txt', robotstart, targetstart, N, epsilon, K)


if __name__ == "__main__":
    # you should change the following line to test different maps
    caught, numofmoves = test_map0()
    print('Number of moves made: {}; Target caught: {}.\n'.format(numofmoves, caught))
    #plt.ioff()
    #plt.show()
