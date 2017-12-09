#!/usr/bin/env python3

from controls import *
from imu import *
from mpu import *

import ev3dev.ev3 as ev3
import os
import smbus
import time

mot_rear = ev3.LargeMotor('outA')
mot_kick = ev3.LargeMotor('outD')
mot_steer = ev3.MediumMotor('outB')

STATE_SETUP_IMU = 0
STATE_SPIN_UP = 1
STATE_RUNNING = 2

# (w, x, y, z)
def cal_lean(quaternion):
    qp = [-quaternion[3], quaternion[2], -quaternion[1], quaternion[0]]
    qinverse = [quaternion[0], -quaternion[1], -quaternion[2],-quaternion[3]]
    pos = [qp[0]*qinverse[1]+qp[1]*qinverse[0]+qp[2]*qinverse[3]-qp[3]*qinverse[2],
           qp[0]*qinverse[2]-qp[1]*qinverse[3]+qp[2]*qinverse[0]+qp[3]*qinverse[1],
           qp[0]*qinverse[3]+qp[1]*qinverse[2]-qp[2]*qinverse[1]+qp[3]*qinverse[0]]
    # print("Position is:",pos)
    # value = math.sqrt(pos[0]**2 + pos[1]**2)
    angle = math.asin(-pos[1])
    # print("Lean angle is:", angle)
    return angle

def main():
    os.nice(-20)
    ev3.Leds.all_off()
    log = []

    # Loop rate
    dT = 0.05

    # Motor speed
    MOT_SPD = 1000
    
    # Push kickstand
    mot_kick.run_direct(duty_cycle_sp=20)
    time.sleep(5)

    # Controls setup
    bp = BikeParams()
    mats = params_to_matrix(bp)
    V = MOT_SPD / 180 * math.pi * bp.r_F * 0.9
    # print(mats)
    mats = mats_at_vel(mats, V)
    # print(mats)

    # x is [Lean, Steer, Lean deriv, Steer Deriv]

    A_, B_ = mats_to_ab(mats)
    Ad, Bd = discretise_time(A_, B_, dT)
    Q = np.array([[10, 0, 0, 0],
                  [0, 5, 0, 0],
                  [0, 0, 10, 0],
                  [0, 0, 0, 1]])
    R = np.array([[1]])
    lqr_Kd = lqr_disc_inf_horiz(Ad, Bd, Q, R)
    print(lqr_Kd)

    # IMU stuff
    imufilt = filterSetup()
    imufilt[0].beta = math.sqrt(3/4) * math.pi * 40 / 180
    imufilt[0].zeta = 0
    imufilt[0].deltat = dT
    b = smbus.SMBus(6)
    setup_mpu(b)
    mag_cal = setup_mag(b)
    print("Magnetometer cal: {}".format(mag_cal))
    gyro_rads_per_s_per_lsb = 250.0 / 180.0 * math.pi / 32768.0

    buttons = ev3.Button()

    # Start state matchine
    state = STATE_SETUP_IMU
    ev3.Leds.set_color(ev3.Leds.LEFT, ev3.Leds.YELLOW)
    remembered_time = time.time()

    try:
        while True:
            time_now = time.time()
            if buttons.backspace:
                break
            if state == STATE_SETUP_IMU:
                # Set up IMU
                accel_data, gyro_data, mag_data, _ = read_mpu_and_mag(b, mag_cal)
                filterUpdate(imufilt,
                    gyro_data[0] * gyro_rads_per_s_per_lsb,
                    gyro_data[1] * gyro_rads_per_s_per_lsb,
                    gyro_data[2] * gyro_rads_per_s_per_lsb,
                    accel_data[0], accel_data[1], accel_data[2],
                    1, 0, 0)
                # print("{} {} {} {}".format(imufilt[0].SEq_1, imufilt[0].SEq_2, imufilt[0].SEq_3, imufilt[0].SEq_4))
                if time.time() - remembered_time >= 10:
                    # Calibrated for 10 seconds
                    mot_steer.position = 0
                    last_steer_pos = 0
                    lean_angle_bias = cal_lean((imufilt[0].SEq_1, imufilt[0].SEq_2, imufilt[0].SEq_3, imufilt[0].SEq_4))
                    #lean_angle_bias = 0
                    ev3.Leds.set_color(ev3.Leds.LEFT, ev3.Leds.GREEN)
                    mot_steer.run_direct(duty_cycle_sp=0)
                    mot_rear.run_forever(speed_sp=MOT_SPD)
                    mot_kick.run_direct(duty_cycle_sp=0)
                    remembered_time = time.time()
                    state = STATE_SPIN_UP
            elif state == STATE_SPIN_UP or state == STATE_RUNNING:
                if state == STATE_SPIN_UP:
                    if time.time() - remembered_time >= 1:
                        mot_kick.run_direct(duty_cycle_sp=-20)
                        state = STATE_RUNNING
                accel_data, gyro_data, mag_data, _ = read_mpu_and_mag(b, mag_cal)
                delta_lean = gyro_data[0] * gyro_rads_per_s_per_lsb
                filterUpdate(imufilt,
                    delta_lean,
                    gyro_data[1] * gyro_rads_per_s_per_lsb,
                    gyro_data[2] * gyro_rads_per_s_per_lsb,
                    accel_data[0], accel_data[1], accel_data[2],
                    1, 0, 0)
                # print("{} {} {} {} {}".format(delta_lean, imufilt[0].SEq_1, imufilt[0].SEq_2, imufilt[0].SEq_3, imufilt[0].SEq_4))

                lean = cal_lean((imufilt[0].SEq_1, imufilt[0].SEq_2, imufilt[0].SEq_3, imufilt[0].SEq_4)) - lean_angle_bias

                steer_deg = mot_steer.position % 360
                if steer_deg > 180:
                    steer_deg -= 360
                #if steer_deg > 90:
                #    steer_deg -= 180
                #if steer_deg < -90:
                #    steer_deg += 180
                steer = steer_deg * math.pi / 180
                # steer_deriv = (steer - last_steer_pos) / dT
                # last_steer_pos = steer

                steer_deriv = mot_steer.speed / mot_steer.count_per_rot * 2 * math.pi

                X = np.array([[lean], [steer], [delta_lean], [steer_deriv]])
                u = -lqr_Kd.dot(X)[0][0]
                # print(X, u)

                # print(u)
                # WTF?
                u = u * 180 / math.pi# * 0.9
                # u = u * 100
                #u = 0

                #log.append((lean, steer, delta_lean, steer_deriv, u))
                log.append(("Input is: ", u))


                duty_cycle = u
                if duty_cycle > 100:
                    duty_cycle = 100
                if duty_cycle < -100:
                    duty_cycle = -100

                mot_steer.duty_cycle_sp = duty_cycle
            else:
                assert False
            # print("Exec took {} ms".format((time.time() - time_now) * 1000))
            elapsed_time = time.time() - time_now
            if elapsed_time <= dT:
                time.sleep(dT - elapsed_time)
    finally:
        try:
            mot_steer.stop()
        except:
            pass
        try:
            mot_rear.stop()
        except:
            pass
        try:
            mot_kick.stop()
        except:
            pass

        f = open('log.txt', 'w')
        f.write(str(log))
        f.close()

if __name__=='__main__':
    main()
