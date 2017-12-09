#!/usr/bin/env python3

import ev3dev.ev3 as ev3
import numpy as np
import scipy.linalg

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
    #Vmax = ev3.PowerSupply().measured_volts
    Vmax = 10
    ktra = 0.007806
    ba = 0.80546
    J = 0.0002873

    M[1][1] += J

    Minv = np.linalg.inv(M)
    # print(Minv)

    negMinvK = -Minv.dot(K)
    negMinvC = -Minv.dot(C)
    negMinvBeta = -Minv.dot(np.array([[0, 0], [0, ba]]))
    negMinvCBeta = negMinvC + negMinvBeta

    A = np.array([[0,   0,      1,  0],
                  [0,   0,      0,  1],
                  [negMinvK[0,0], negMinvK[0,1],    negMinvCBeta[0,0], negMinvCBeta[0,1]],
                  [negMinvK[1,0], negMinvK[1,1],    negMinvCBeta[1,0], negMinvCBeta[1,1]]])

    # print(negMinvK)
    # print(negMinvC)
    # print(A)

    MinvOne = Minv.dot(np.array([[0],[1]]))

    B = np.array([[0], [0], [MinvOne[0]], [MinvOne[1]]]) * Vmax * ktra

    # print(MinvOne)
    # print(B)

    return (A, B)

# Borrowed from http://www.mwm.im/lqr-controllers-with-python/
def lqr_cont_inf_horiz(A, B, Q, R):
    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_continuous_are(A, B, Q, R)
     
    #compute the LQR gain
    K = scipy.linalg.inv(R).dot((B.T.dot(X)))
     
    return K

# Stolen from controlpy
def lqr_disc_inf_horiz(A, B, Q, R):
    #first, try to solve the ricatti equation
    X = scipy.linalg.solve_discrete_are(A, B, Q, R)
    
    #compute the LQR gain
    K = np.dot(np.linalg.inv(np.dot(np.dot(B.T,X),B)+R),(np.dot(np.dot(B.T,X),A)))
    
    return K

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

def demo():
    V = 0.0

    mats = params_to_matrix(BikeParams())
    print(mats)
    mats = mats_at_vel(mats, V)
    print(mats)

    # x is [Lean, Steer, Lean deriv, Steer Deriv]

    ##### Stuff for LQR
    A_, B_ = mats_to_ab(mats)
    Q = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    R = np.array([[1]])
    lqr_K = lqr_cont_inf_horiz(A_, B_, Q, R)
    print(lqr_K)

if __name__=='__main__':
    demo()
