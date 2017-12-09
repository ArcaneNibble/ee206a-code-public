#!/usr/bin/env python3

import numpy as np

##### Model parameters
g = 9.81

w = 1.02
c = 0.08
lambda_ = np.pi / 10

r_R = 0.3
m_R = 2
I_R_xx = 0.0603
I_R_yy = 0.12

x_B = 0.3
z_B = -0.9
m_B = 85
I_B_xx = 9.2
I_B_xz = 2.4
I_B_yy = 11
I_B_zz = 2.8

x_H = 0.9
z_H = -0.7
m_H = 4
I_H_xx = 0.05892
I_H_xz = -0.00756
I_H_yy = 0.06
I_H_zz = 0.00708

r_F = 0.35
m_F = 3
I_F_xx = 0.1405
I_F_yy = 0.28

##### Computations
cos_l = np.cos(lambda_)
sin_l = np.sin(lambda_)

m_T = m_R + m_B + m_H + m_F
x_T = (x_B*m_B + x_H*m_H + w*m_F) / m_T
z_T = (-r_R*m_R + z_B*m_B + z_H*m_H - r_F*m_F) / m_T

I_T_xx = I_R_xx + I_B_xx + I_H_xx + I_F_xx + m_R*r_R**2 + m_B*z_B**2+m_H*z_H**2+m_F*r_F**2
I_T_xz = I_B_xz + I_H_xz - m_B*x_B*z_B - m_H*x_H*z_H + m_F*w*r_F

I_R_zz = I_R_xx
I_F_zz = I_F_xx

I_T_zz = I_R_zz + I_B_zz + I_H_zz + I_F_zz + m_B*x_B**2 + m_H*x_H**2 + m_F*w**2

m_A = m_H + m_F
x_A = (x_H*m_H + w*m_F)/m_A
z_A = (z_H*m_H - r_F*m_F)/m_A

I_A_xx = I_H_xx + I_F_xx + m_H*(z_H - z_A)**2 + m_F*(r_F + z_A)**2
I_A_xz = I_H_xz - m_H*(x_H - x_A)*(z_H - z_A) + m_F*(w - x_A)*(r_F + z_A)
I_A_zz = I_H_zz + I_F_zz + m_H*(x_H - x_A)**2 + m_F*(w - x_A)**2

u_A = (x_A - w - c)*cos_l - z_A*sin_l

I_A_lambdalambda = m_A*u_A**2 + I_A_xx*sin_l**2 + 2*I_A_xz*sin_l*cos_l + I_A_zz*cos_l**2
I_A_lambdax = -m_A*u_A*z_A + I_A_xx*sin_l + I_A_xz*cos_l
I_A_lambdaz = m_A*u_A*x_A + I_A_xz*sin_l + I_A_zz*cos_l

mu = c/w*cos_l

S_R = I_R_yy / r_R
S_F = I_F_yy / r_F
S_T = S_R + S_F

S_A = m_A*u_A + mu*m_T*x_T

M_phiphi = I_T_xx
# print(M_phiphi)
M_phidelta = I_A_lambdax + mu*I_T_xz
# print(M_phidelta)
M_deltadelta = I_A_lambdalambda + 2*mu*I_A_lambdaz + mu**2*I_T_zz
# print(M_deltadelta)

K_0_phiphi = m_T*z_T
# print(K_0_phiphi)
K_0_phidelta = -S_A
# print(K_0_phidelta)
K_0_deltadelta = -S_A*sin_l
# print(K_0_deltadelta)

K_2_phidelta = (S_T - m_T*z_T)/w*cos_l
# print(K_2_phidelta)
K_2_deltadelta = (S_A + S_F*sin_l)/w*cos_l
# print(K_2_deltadelta)

C_1_phidelta = mu*S_T + S_F*cos_l + I_T_xz/w*cos_l - mu*m_T*z_T
# print(C_1_phidelta)
C_1_deltaphi = -(mu*S_T + S_F*cos_l)
# print(C_1_deltaphi)
C_1_deltadelta = I_A_lambdaz/w*cos_l + mu*(S_A + I_T_zz/w*cos_l)
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
print(M)
print(K_0)
print(K_2)
print(C_1)
