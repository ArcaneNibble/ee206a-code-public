#!/usr/bin/env python3

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import random

g = 9.81

##### Model parameters
class BikeParams:
    w = 0.221
    c = 0.0237
    lambda_ = np.pi / 6

    r_R = 0.041
    m_R = 0.0346
    I_R_xx = 2.10368e-05
    I_R_yy = 4.07761e-05

    x_B = 0.0768
    z_B = -0.1005
    m_B = 0.6824
    I_B_xx = 0.0012
    I_B_xz = 6.519389010215833e-04
    I_B_yy = 0.0021
    I_B_zz = 0.0019

    x_H = 0.207
    z_H = -0.09
    m_H = 0.016
    I_H_xx = 1.263e-05
    I_H_xz = -3.866630222816761e-06
    I_H_yy = 1.41456e-05
    I_H_zz = 8.1652e-06

    r_F = 0.041
    m_F = 0.031
    I_F_xx = 1.910940833333334e-05
    I_F_yy = 3.662645e-05

def params_to_matrix(p):
    ##### Computations
    cos_l = np.cos(p.lambda_)
    sin_l = np.sin(p.lambda_)

    m_T = p.m_R + p.m_B + p.m_H + p.m_F
    x_T = (p.x_B*p.m_B + p.x_H*p.m_H + p.w*p.m_F) / m_T
    z_T = (-p.r_R*p.m_R + p.z_B*p.m_B + p.z_H*p.m_H - p.r_F*p.m_F) / m_T

    I_T_xx = p.I_R_xx + p.I_B_xx + p.I_H_xx + p.I_F_xx + p.m_R*p.r_R**2 + p.m_B*p.z_B**2+p.m_H*p.z_H**2+p.m_F*p.r_F**2
    I_T_xz = p.I_B_xz + p.I_H_xz - p.m_B*p.x_B*p.z_B - p.m_H*p.x_H*p.z_H + p.m_F*p.w*p.r_F

    I_R_zz = p.I_R_xx
    I_F_zz = p.I_F_xx

    I_T_zz = I_R_zz + p.I_B_zz + p.I_H_zz + I_F_zz + p.m_B*p.x_B**2 + p.m_H*p.x_H**2 + p.m_F*p.w**2

    m_A = p.m_H + p.m_F
    x_A = (p.x_H*p.m_H + p.w*p.m_F)/m_A
    z_A = (p.z_H*p.m_H - p.r_F*p.m_F)/m_A

    I_A_xx = p.I_H_xx + p.I_F_xx + p.m_H*(p.z_H - z_A)**2 + p.m_F*(p.r_F + z_A)**2
    I_A_xz = p.I_H_xz - p.m_H*(p.x_H - x_A)*(p.z_H - z_A) + p.m_F*(p.w - x_A)*(p.r_F + z_A)
    I_A_zz = p.I_H_zz + I_F_zz + p.m_H*(p.x_H - x_A)**2 + p.m_F*(p.w - x_A)**2

    u_A = (x_A - p.w - p.c)*cos_l - z_A*sin_l

    I_A_lambdalambda = m_A*u_A**2 + I_A_xx*sin_l**2 + 2*I_A_xz*sin_l*cos_l + I_A_zz*cos_l**2
    I_A_lambdax = -m_A*u_A*z_A + I_A_xx*sin_l + I_A_xz*cos_l
    I_A_lambdaz = m_A*u_A*x_A + I_A_xz*sin_l + I_A_zz*cos_l

    mu = p.c/p.w*cos_l

    S_R = p.I_R_yy / p.r_R
    S_F = p.I_F_yy / p.r_F
    S_T = S_R + S_F

    S_A = m_A*u_A + mu*m_T*x_T

    M_phiphi = I_T_xx
    # print(p.M_phiphi)
    M_phidelta = I_A_lambdax + mu*I_T_xz
    # print(p.M_phidelta)
    M_deltadelta = I_A_lambdalambda + 2*mu*I_A_lambdaz + mu**2*I_T_zz
    # print(p.M_deltadelta)

    K_0_phiphi = m_T*z_T
    # print(K_0_phiphi)
    K_0_phidelta = -S_A
    # print(K_0_phidelta)
    K_0_deltadelta = -S_A*sin_l
    # print(K_0_deltadelta)

    K_2_phidelta = (S_T - m_T*z_T)/p.w*cos_l
    # print(K_2_phidelta)
    K_2_deltadelta = (S_A + S_F*sin_l)/p.w*cos_l
    # print(K_2_deltadelta)

    C_1_phidelta = mu*S_T + S_F*cos_l + I_T_xz/p.w*cos_l - mu*m_T*z_T
    # print(C_1_phidelta)
    C_1_deltaphi = -(mu*S_T + S_F*cos_l)
    # print(C_1_deltaphi)
    C_1_deltadelta = I_A_lambdaz/p.w*cos_l + mu*(S_A + I_T_zz/p.w*cos_l)
    # print(C_1_deltadelta)

    ##### Assemble output
    M = np.array([[M_phiphi, M_phidelta],
                  [M_phidelta, M_deltadelta]])
    K_0 = np.array([[K_0_phiphi, K_0_phidelta],
                    [K_0_phidelta, K_0_deltadelta]])
    K_2 = np.array([[0, K_2_phidelta],
                    [0, K_2_deltadelta]])
    C_1 = np.array([[0, C_1_phidelta],
                    [C_1_deltaphi, C_1_deltadelta]])

    return (M, K_0, K_2, C_1)

def mats_at_vel(mats, v):
    M, K_0, K_2, C_1 = mats

    C = v * C_1
    K = g * K_0 + v**2*K_2

    return (M, C, K)

def mats_to_ab(mats):
    M, C, K = mats

    # Motor params
    Vmax = 10
    ktra = 0.007806
    ba = 0.006689
    J = 0.0002873

    M[1][1] += J

    Minv = np.linalg.inv(M)
    # print(Minv)

    negMinvK = -Minv @ K
    negMinvC = -Minv @ C
    negMinvBeta = -Minv @ np.array([[0, 0], [0, ba]])
    negMinvCBeta = negMinvC + negMinvBeta

    A = np.array([[0,   0,      1,  0],
                  [0,   0,      0,  1],
                  [negMinvK[0,0], negMinvK[0,1],    negMinvCBeta[0,0], negMinvCBeta[0,1]],
                  [negMinvK[1,0], negMinvK[1,1],    negMinvCBeta[1,0], negMinvCBeta[1,1]]])

    # print(negMinvK)
    # print(negMinvC)
    # print(A)

    MinvOne = Minv @ np.array([[0],[1]])

    B = np.array([[0], [0], [MinvOne[0]], [MinvOne[1]]]) * Vmax * ktra

    # print(MinvOne)
    # print(B)

    return (A, B)

# Borrowed from http://www.mwm.im/lqr-controllers-with-python/
def lqr_cont_inf_horiz(A, B, Q, R):
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K

# Stolen from controlpy
def lqr_disc_inf_horiz(A, B, Q, R):
    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
    #compute the LQR gain
    K = np.dot(np.linalg.inv(np.dot(np.dot(B.T,X),B)+R),(np.dot(np.dot(B.T,X),A)))
    
    return K

def sim_cont(mats, x0, feedback_fn, timestep, num_steps):
    ##### Convert to ABCD/first-order form
    A, B = mats

    new_x0 = np.array([[x0[0]], [x0[1]], [0], [0]])
    # print(new_x0)

    ##### The x' = f(t, x) equation for the ODE
    def f(t, x):
        # print(t,  x)

        x = np.array([[x[0]], [x[1]], [x[2]], [x[3]]])

        u = feedback_fn(t, x)
        u = np.array([[u]])

        xdot = A @ x + B @ u
        # print(xdot)

        return xdot

    ##### Set up the integration
    r = scipy.integrate.ode(f)
    r.set_initial_value(new_x0)

    ##### Do the integration
    aborted = False
    result_x = []
    result_y0 = []
    result_y1 = []
    result_y2 = []
    result_y3 = []
    while r.successful() and r.t < num_steps * timestep:
        if np.abs(r.y[0]) > np.pi / 2 or np.abs(r.y[1]) > np.pi / 4:
            # print("Bike fell over, aborting")
            aborted = True
            break

        r.integrate(r.t+timestep)
        # print(r.t, r.y)
        result_x.append(r.t)
        result_y0.append(r.y[0])
        result_y1.append(r.y[1])
        result_y2.append(r.y[2])
        result_y3.append(r.y[3])

    return (result_x, result_y0, result_y1, result_y2, result_y3, aborted)

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


# Stolen from controlpy
def discretise_time(A, B, dt):
    nstates = A.shape[0]
    ninputs = B.shape[1]

    M = np.matrix(np.zeros([nstates+ninputs,nstates+ninputs]))
    M[:nstates,:nstates] = A
    M[:nstates, nstates:] = B
    
    Md = scipy.linalg.expm(M*dt)
    Ad = Md[:nstates, :nstates]
    Bd = Md[:nstates, nstates:]

    return Ad, Bd

##### Actual main stuff
def main():
    V = 1000 / 180 * np.pi * 0.041
    # V = 0.9*0.59
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
    lqr_K = lqr_cont_inf_horiz(A_, B_, Q, R)
    lqr_Kd = lqr_disc_inf_horiz(Ad, Bd, Q, R)
    print(lqr_K)
    print(lqr_Kd)

    def lqr_feedback_fn(t, y):
        return -lqr_K @ y

    def lqrd_feedback_fn(t, y):
        return (-lqr_Kd @ y)[0][0]

    # result_x, result_y0, result_y1, result_y2, result_y3, aborted = sim_cont((A_, B_), [0.1, 0], lqr_feedback_fn, deltaT, 20000)
    # if aborted:
    #     print("Simulation aborted because bike fell over or steered too much")

    # plt.plot(result_x, result_y0, label="Lean angle")
    # plt.plot(result_x, result_y1, label="Steer angle")
    # plt.plot(result_x, result_y2, label="Lean angle derivative")
    # plt.plot(result_x, result_y3, label="Steer angle derivative")
    # plt.legend()
    # plt.show()

    result_x, result_u, result_y0, result_y1, result_y2, result_y3, aborted = sim_disc((Ad, Bd), [0.1, 0], lqrd_feedback_fn, deltaT, 20000)
    if aborted:
        print("Simulation aborted because bike fell over or steered too much")

    plt.plot(result_x, result_u, label="Discrete Control output")
    plt.plot(result_x, result_y0, label="Discrete Lean angle")
    plt.plot(result_x, result_y1, label="Discrete Steer angle")
    # plt.plot(result_x, result_y2, label="Discrete Lean angle derivative")
    # plt.plot(result_x, result_y3, label="Discrete Steer angle derivative")
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
