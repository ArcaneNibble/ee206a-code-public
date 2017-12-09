#include <math.h>
#include <stdlib.h>

typedef struct filt_state {
    // System constants
    float deltat;   // sampling period in seconds (shown as 1 ms)
    float beta;
    float zeta;

    // The estimated quaternion
    float SEq_1, SEq_2, SEq_3, SEq_4;               // estimated orientation quaternion elements with initial conditions
    // The estimated velocity
    float SEqDot_1, SEqDot_2, SEqDot_3, SEqDot_4;

    // Global system variables
    float b_x, b_z;                                 // reference direction of flux in earth frame
    float w_bx, w_by, w_bz;                         // estimate gyroscope biases error
} filt_state;

filt_state *filterSetup() {
    filt_state *ret = malloc(sizeof(filt_state));

    ret->deltat = 0.001f;
    const float gyroMeasError = 3.14159265358979f * (5.0f / 180.0f);    // gyroscope measurement error in rad/s (shown as 5 deg/s)
    const float gyroMeasDrift = 3.14159265358979f * (0.2f / 180.0f);    // gyroscope measurement error in rad/s/s (shown as 0.2f deg/s/s)
    ret->beta = sqrtf(3.0f / 4.0f) * gyroMeasError;
    ret->zeta = sqrtf(3.0f / 4.0f) * gyroMeasDrift;

    ret->SEq_1 = 1;
    ret->SEq_2 = 0;
    ret->SEq_3 = 0;
    ret->SEq_4 = 0;

    ret->b_x = 1;
    ret->b_z = 0;

    ret->w_bx = 0;
    ret->w_by = 0;
    ret->w_bz = 0;

    return ret;
}

// Function to compute one filter iteration
void filterUpdate(filt_state *s, float w_x, float w_y, float w_z, float a_x, float a_y, float a_z, float m_x, float m_y, float m_z)
{
    // local system variables
    float norm;                                                             // vector norm
    float SEqDot_omega_1, SEqDot_omega_2, SEqDot_omega_3, SEqDot_omega_4;   // quaternion rate from gyroscopes elements
    float f_1, f_2, f_3, f_4, f_5, f_6;                                     // objective function elements
    float J_11or24, J_12or23, J_13or22, J_14or21, J_32, J_33,               // objective function Jacobian elements
    J_41, J_42, J_43, J_44, J_51, J_52, J_53, J_54, J_61, J_62, J_63, J_64; //
    float SEqHatDot_1, SEqHatDot_2, SEqHatDot_3, SEqHatDot_4;               // estimated direction of the gyroscope error
    float w_err_x, w_err_y, w_err_z;                                        // estimated direction of the gyroscope error (angular)
    float h_x, h_y, h_z;                                                    // computed flux in the earth frame

    // axulirary variables to avoid reapeated calcualtions
    float halfSEq_1 = 0.5f * s->SEq_1;
    float halfSEq_2 = 0.5f * s->SEq_2;
    float halfSEq_3 = 0.5f * s->SEq_3;
    float halfSEq_4 = 0.5f * s->SEq_4;
    float twoSEq_1 = 2.0f * s->SEq_1;
    float twoSEq_2 = 2.0f * s->SEq_2;
    float twoSEq_3 = 2.0f * s->SEq_3;
    float twoSEq_4 = 2.0f * s->SEq_4;
    float twob_x = 2.0f * s->b_x;
    float twob_z = 2.0f * s->b_z;
    float twob_xSEq_1 = 2.0f * s->b_x * s->SEq_1;
    float twob_xSEq_2 = 2.0f * s->b_x * s->SEq_2;
    float twob_xSEq_3 = 2.0f * s->b_x * s->SEq_3;
    float twob_xSEq_4 = 2.0f * s->b_x * s->SEq_4;
    float twob_zSEq_1 = 2.0f * s->b_z * s->SEq_1;
    float twob_zSEq_2 = 2.0f * s->b_z * s->SEq_2;
    float twob_zSEq_3 = 2.0f * s->b_z * s->SEq_3;
    float twob_zSEq_4 = 2.0f * s->b_z * s->SEq_4;
    float SEq_1SEq_2;
    float SEq_1SEq_3 = s->SEq_1 * s->SEq_3;
    float SEq_1SEq_4;
    float SEq_2SEq_3;
    float SEq_2SEq_4 = s->SEq_2 * s->SEq_4;
    float SEq_3SEq_4;
    float twom_x = 2.0f * m_x;
    float twom_y = 2.0f * m_y;
    float twom_z = 2.0f * m_z;

    // normalise the accelerometer measurement
    norm = sqrtf(a_x * a_x + a_y * a_y + a_z * a_z);
    a_x /= norm;
    a_y /= norm;
    a_z /= norm;

    // normalise the magnetometer measurement
    norm = sqrtf(m_x * m_x + m_y * m_y + m_z * m_z);
    m_x /= norm;
    m_y /= norm;
    m_z /= norm;

    // compute the objective function and Jacobian
    f_1 = twoSEq_2 * s->SEq_4 - twoSEq_1 * s->SEq_3 - a_x;
    f_2 = twoSEq_1 * s->SEq_2 + twoSEq_3 * s->SEq_4 - a_y;
    f_3 = 1.0f - twoSEq_2 * s->SEq_2 - twoSEq_3 * s->SEq_3 - a_z;
    f_4 = twob_x * (0.5f - s->SEq_3 * s->SEq_3 - s->SEq_4 * s->SEq_4) + twob_z * (SEq_2SEq_4 - SEq_1SEq_3) - m_x;
    f_5 = twob_x * (s->SEq_2 * s->SEq_3 - s->SEq_1 * s->SEq_4) + twob_z * (s->SEq_1 * s->SEq_2 + s->SEq_3 * s->SEq_4) - m_y;
    f_6 = twob_x * (SEq_1SEq_3 + SEq_2SEq_4) + twob_z * (0.5f - s->SEq_2 * s->SEq_2 - s->SEq_3 * s->SEq_3) - m_z;
    J_11or24 = twoSEq_3;                                                    // J_11 negated in matrix multiplication
    J_12or23 = 2.0f * s->SEq_4;
    J_13or22 = twoSEq_1;                                                    // J_12 negated in matrix multiplication
    J_14or21 = twoSEq_2;
    J_32 = 2.0f * J_14or21;                                                 // negated in matrix multiplication
    J_33 = 2.0f * J_11or24;                                                 // negated in matrix multiplication
    J_41 = twob_zSEq_3;                                                     // negated in matrix multiplication
    J_42 = twob_zSEq_4;
    J_43 = 2.0f * twob_xSEq_3 + twob_zSEq_1;                                // negated in matrix multiplication
    J_44 = 2.0f * twob_xSEq_4 - twob_zSEq_2;                                // negated in matrix multiplication
    J_51 = twob_xSEq_4 - twob_zSEq_2;                                       // negated in matrix multiplication
    J_52 = twob_xSEq_3 + twob_zSEq_1;
    J_53 = twob_xSEq_2 + twob_zSEq_4;
    J_54 = twob_xSEq_1 - twob_zSEq_3;                                       // negated in matrix multiplication
    J_61 = twob_xSEq_3;
    J_62 = twob_xSEq_4 - 2.0f * twob_zSEq_2;
    J_63 = twob_xSEq_1 - 2.0f * twob_zSEq_3;
    J_64 = twob_xSEq_2;

    // compute the gradient (matrix multiplication)
    SEqHatDot_1 = J_14or21 * f_2 - J_11or24 * f_1 - J_41 * f_4 - J_51 * f_5 + J_61 * f_6;
    SEqHatDot_2 = J_12or23 * f_1 + J_13or22 * f_2 - J_32 * f_3 + J_42 * f_4 + J_52 * f_5 + J_62 * f_6;
    SEqHatDot_3 = J_12or23 * f_2 - J_33 * f_3 - J_13or22 * f_1 - J_43 * f_4 + J_53 * f_5 + J_63 * f_6;
    SEqHatDot_4 = J_14or21 * f_1 + J_11or24 * f_2 - J_44 * f_4 - J_54 * f_5 + J_64 * f_6;

    // normalise the gradient to estimate direction of the gyroscope error
    norm = sqrtf(SEqHatDot_1 * SEqHatDot_1 + SEqHatDot_2 * SEqHatDot_2 + SEqHatDot_3 * SEqHatDot_3 + SEqHatDot_4 * SEqHatDot_4);
    SEqHatDot_1 = SEqHatDot_1 / norm;
    SEqHatDot_2 = SEqHatDot_2 / norm;
    SEqHatDot_3 = SEqHatDot_3 / norm;
    SEqHatDot_4 = SEqHatDot_4 / norm;

    // compute angular estimated direction of the gyroscope error
    w_err_x = twoSEq_1 * SEqHatDot_2 - twoSEq_2 * SEqHatDot_1 - twoSEq_3 * SEqHatDot_4 + twoSEq_4 * SEqHatDot_3;
    w_err_y = twoSEq_1 * SEqHatDot_3 + twoSEq_2 * SEqHatDot_4 - twoSEq_3 * SEqHatDot_1 - twoSEq_4 * SEqHatDot_2;
    w_err_z = twoSEq_1 * SEqHatDot_4 - twoSEq_2 * SEqHatDot_3 + twoSEq_3 * SEqHatDot_2 - twoSEq_4 * SEqHatDot_1;

    // compute and remove the gyroscope baises
    s->w_bx += w_err_x * s->deltat * s->zeta;
    s->w_by += w_err_y * s->deltat * s->zeta;
    s->w_bz += w_err_z * s->deltat * s->zeta;
    w_x -= s->w_bx;
    w_y -= s->w_by;
    w_z -= s->w_bz;

    // compute the quaternion rate measured by gyroscopes
    SEqDot_omega_1 = -halfSEq_2 * w_x - halfSEq_3 * w_y - halfSEq_4 * w_z;
    SEqDot_omega_2 = halfSEq_1 * w_x + halfSEq_3 * w_z - halfSEq_4 * w_y;
    SEqDot_omega_3 = halfSEq_1 * w_y - halfSEq_2 * w_z + halfSEq_4 * w_x;
    SEqDot_omega_4 = halfSEq_1 * w_z + halfSEq_2 * w_y - halfSEq_3 * w_x;

    // compute then integrate the estimated quaternion rate
    s->SEqDot_1 = (SEqDot_omega_1 - (s->beta * SEqHatDot_1));
    s->SEqDot_2 = (SEqDot_omega_2 - (s->beta * SEqHatDot_2));
    s->SEqDot_3 = (SEqDot_omega_3 - (s->beta * SEqHatDot_3));
    s->SEqDot_4 = (SEqDot_omega_4 - (s->beta * SEqHatDot_4));
    s->SEq_1 += s->SEqDot_1 * s->deltat;
    s->SEq_2 += s->SEqDot_2 * s->deltat;
    s->SEq_3 += s->SEqDot_3 * s->deltat;
    s->SEq_4 += s->SEqDot_4 * s->deltat;

    // normalise quaternion
    norm = sqrtf(s->SEq_1 * s->SEq_1 + s->SEq_2 * s->SEq_2 + s->SEq_3 * s->SEq_3 + s->SEq_4 * s->SEq_4);
    s->SEq_1 /= norm;
    s->SEq_2 /= norm;
    s->SEq_3 /= norm;
    s->SEq_4 /= norm;

    // compute flux in the earth frame
    SEq_1SEq_2 = s->SEq_1 * s->SEq_2;                                       // recompute axulirary variables
    SEq_1SEq_3 = s->SEq_1 * s->SEq_3;
    SEq_1SEq_4 = s->SEq_1 * s->SEq_4;
    SEq_3SEq_4 = s->SEq_3 * s->SEq_4;
    SEq_2SEq_3 = s->SEq_2 * s->SEq_3;
    SEq_2SEq_4 = s->SEq_2 * s->SEq_4;
    h_x = twom_x * (0.5f - s->SEq_3 * s->SEq_3 - s->SEq_4 * s->SEq_4) + twom_y * (SEq_2SEq_3 - SEq_1SEq_4) + twom_z * (SEq_2SEq_4 + SEq_1SEq_3);
    h_y = twom_x * (SEq_2SEq_3 + SEq_1SEq_4) + twom_y * (0.5f - s->SEq_2 * s->SEq_2 - s->SEq_4 * s->SEq_4) + twom_z * (SEq_3SEq_4 - SEq_1SEq_2);
    h_z = twom_x * (SEq_2SEq_4 - SEq_1SEq_3) + twom_y * (SEq_3SEq_4 + SEq_1SEq_2) + twom_z * (0.5f - s->SEq_2 * s->SEq_2 - s->SEq_3 * s->SEq_3);

    // normalise the flux vector to have only components in the x and z
    s->b_x = sqrtf((h_x * h_x) + (h_y * h_y));
    s->b_z = h_z;
}
