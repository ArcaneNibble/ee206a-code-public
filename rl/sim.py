#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import random

from controls import *

def sim_disc(mats, x0, feedback_fn, timestep, num_steps):
    ##### Convert to ABCD/first-order form
    A, B = mats

    x_t_1 = np.array([[x0[0]], [x0[1]], [0], [0]])

    ##### Do the sim
    aborted = False
    result_x = [0]
    result_u = [0]
    result_y0 = [x_t_1[0]]
    result_y1 = [x_t_1[1]]
    result_y2 = [x_t_1[2]]
    result_y3 = [x_t_1[3]]
    t = 0
    while t < num_steps:
        if np.abs(x_t_1[0]) > np.pi / 2 or np.abs(x_t_1[1]) > np.pi / 4:
            # print("Bike fell over, aborting")
            aborted = True
            break

        u = feedback_fn(t, x_t_1)
        u = np.array([[u]])
        # print(u)
        x_t = A @ x_t_1 + B @ u
        # print(u, x_t)

        t += 1
        x_t_1 = x_t

        result_x.append(t * timestep)
        result_u.append(u)
        result_y0.append(x_t[0])
        result_y1.append(x_t[1])
        result_y2.append(x_t[2])
        result_y3.append(x_t[3])

    return (result_x, result_u, result_y0, result_y1, result_y2, result_y3, aborted)

##### Actual main stuff
def main():
    # V = 1000 / 180 * np.pi * 0.041
    # V = 0.9*0.59
    V = 3.5
    deltaT = 0.05

    mats = params_to_matrix(BikeParams())
    mats = mats_at_vel(mats, V)
    print(mats)

    def zero_feedback_fn(t, y):
        # print(t, y)
        return 0

    ##### Stuff for LQR
    A_, B_ = mats_to_ab(mats)
    Ad, Bd = discretise_time(A_, B_, deltaT)
    print(A_)
    print(B_)
    print(Ad)
    print(Bd)
    Q = np.array([[10, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 10, 0],
                  [0, 0, 0, 1]])
    R = np.array([[8]])
    lqr_Kd = lqr_disc_inf_horiz(Ad, Bd, Q, R)
    print(lqr_Kd)

    def lqrd_feedback_fn(t, y):
        return (-lqr_Kd @ y)[0][0]

    result_x, result_u, result_y0, result_y1, result_y2, result_y3, aborted = sim_disc((Ad, Bd), [0.1, 0], zero_feedback_fn, deltaT, 10000)
    if aborted:
        print("Simulation aborted because bike fell over or steered too much")

    plt.plot(result_x, result_u, label="Discrete Control output")
    plt.plot(result_x, result_y0, label="Discrete Lean angle")
    plt.plot(result_x, result_y1, label="Discrete Steer angle")
    plt.plot(result_x, result_y2, label="Discrete Lean angle derivative")
    plt.plot(result_x, result_y3, label="Discrete Steer angle derivative")
    plt.legend()
    plt.show()

def main2():
    bp = BikeParams()

    V = 1000 / 180 * np.pi * bp.r_F
    deltaT = 0.05

    mats = params_to_matrix(bp)
    mats = mats_at_vel(mats, V)
    A_, B_ = mats_to_ab(mats)
    Ad, Bd = discretise_time(A_, B_, deltaT)

    ##### Stuff for RL
    alpha = 0.000000001
    gamma = 0.5
    first = True
    Ws2 = np.zeros((4, 4))
    # Ws2 = Ws2.T.dot(Ws2)
    Wa2 = .001
    Was = np.zeros((1, 4))
    Wa = 0
    Ws = np.zeros((1, 4))
    Wc = 0
    last_x = None
    last_a = None
    last_Q = None

    def rl_feedback_fn(t, x):
        nonlocal first, Ws2, Wa2, Was, Wa, Ws, Wc, last_x, last_a, last_Q
        # Pick the best action to do now
        a = -0.5 / Wa2 * (Was.dot(x)[0][0] + Wa)
        a += np.random.uniform(-5, 5)
        # a = np.random.uniform(-100,100)
        # print("a", a)
        # Calculate what Q we think we will get
        Q = x.T.dot(Ws2.dot(x)) + Wa2*a**2 + a*Was.dot(x) + Wa*a + Ws.dot(x) + Wc
        Q = Q[0][0]
        # print("Q", Q)

        if not first:
            # Compute reward for last time
            R = 2 - 0.5*x[0]**2 - 0.5*x[1]**2
            R = R[0]
            # R = 1
            # print("R", R)

            # Update the weight matrices for last time
            err = alpha*(R + gamma*Q - last_Q)
            # print("err", err)
            # print(last_x.dot(last_x.T))
            Ws2 += err*last_x.dot(last_x.T)
            Wa2 += err*last_a**2
            Was += err*last_a*last_x.T
            Wa += err*last_a
            Ws += err*last_x.T
            Wc += err
            # print(Ws2)
            # print(Wa2)
            # print(Was)
            # print(Wa)
            # print(Ws)
            # print(Wc)

            # new_last_Q = last_x.T.dot(Ws2.dot(last_x)) + Wa2*last_a**2 + last_a*Was.dot(last_x) + Wa*last_a + Ws.dot(last_x) + Wc
            # new_last_Q = new_last_Q[0][0]
            # print("new last Q", new_last_Q)
            # if new_last_Q > last_Q:
            #     print("WTFWTFWTF")

        last_x = x
        last_a = a
        last_Q = Q

        first = False
        return a

    i = 0
    try:
        while True:
            initial_lean_angle = np.random.normal(0, 0.2)
            # initial_lean_angle = 0.001
            first = True
            result_x, result_u, result_y0, result_y1, result_y2, result_y3, aborted = sim_disc((Ad, Bd), [initial_lean_angle, 0], rl_feedback_fn, deltaT, 10000)

            avg_cost = sum(((0.5*x**2) + (0.5*y**2) for x, y in zip(result_y0, result_y1))) / len(result_x)

            print("Iter {}, aborted? {}, steps {}, avg cost {}".format(i, aborted, len(result_x), avg_cost[0]))

            i = i + 1

    except KeyboardInterrupt:
        pass

    print("*" * 80)
    print(Ws2)
    print(Wa2)
    print(Was)
    print(Wa)
    print(Ws)
    print(Wc)

if __name__=='__main__':
    main2()
