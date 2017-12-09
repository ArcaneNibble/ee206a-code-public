#!/usr/bin/env python3

from imu import *

import math
import os
import smbus
import socket
import struct
import time

MPU_ADDR = 0x68
MAG_ADDR = 0x0C

SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
ACCEL_CONFIG = 0x1C
ACCEL_CONFIG2 = 0x1D
I2C_MST_CTRL = 0x24   
I2C_SLV0_ADDR = 0x25
I2C_SLV0_REG = 0x26
I2C_SLV0_CTRL = 0x27
EXT_SENS_DATA_00 = 0x49
I2C_SLV0_DO = 0x63
USER_CTRL = 0x6A
PWR_MGMT_1 = 0x6B

MAG_ST1 = 0x02
MAG_CTRL = 0x0A
MAG_ASAX = 0x10
MAG_ASAY = 0x11
MAG_ASAZ = 0x12

def mag_reg_rd(b, reg):
    b.write_byte_data(MPU_ADDR, I2C_SLV0_ADDR, 0x80 | MAG_ADDR)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_REG, reg)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_CTRL, 0x81)
    time.sleep(0.5)
    return b.read_byte_data(MPU_ADDR, EXT_SENS_DATA_00)

def mag_reg_wr(b, reg, val):
    b.write_byte_data(MPU_ADDR, I2C_SLV0_ADDR, MAG_ADDR)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_REG, reg)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_CTRL, 0x81)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_DO, val)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_CTRL, 0x81)
    time.sleep(0.5)

def setup_mpu(b):
    # Wake up
    b.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x00)
    time.sleep(0.1)
    b.write_byte_data(MPU_ADDR, PWR_MGMT_1, 0x01)
    time.sleep(0.2)

    b.write_byte_data(MPU_ADDR, CONFIG, 0x03)
    # 100 Hz
    b.write_byte_data(MPU_ADDR, SMPLRT_DIV, 0x09)

    # 250dps
    b.write_byte_data(MPU_ADDR, GYRO_CONFIG, 0x00)
    # 2g
    b.write_byte_data(MPU_ADDR, ACCEL_CONFIG, 0x00)
    b.write_byte_data(MPU_ADDR, ACCEL_CONFIG2, 0x03)

    b.write_byte_data(MPU_ADDR, USER_CTRL, 0x20)
    b.write_byte_data(MPU_ADDR, I2C_MST_CTRL, 0x0D)

def setup_mag(b):
    # Read calibration
    mag_reg_wr(b, MAG_CTRL, 0x00)
    mag_reg_wr(b, MAG_CTRL, 0x0F)
    cal_x = mag_reg_rd(b, MAG_ASAX)
    cal_y = mag_reg_rd(b, MAG_ASAY)
    cal_z = mag_reg_rd(b, MAG_ASAZ)
    mag_reg_wr(b, MAG_CTRL, 0x00)

    mag_reg_wr(b, MAG_CTRL, 0x12)

    # Set up auto poll
    b.write_byte_data(MPU_ADDR, I2C_SLV0_ADDR, 0x80 | MAG_ADDR)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_REG, MAG_ST1)
    b.write_byte_data(MPU_ADDR, I2C_SLV0_CTRL, 0x88)

    # Manipulate calibration
    cal_x = (cal_x - 128) / 256.0 + 1
    cal_y = (cal_y - 128) / 256.0 + 1
    cal_z = (cal_z - 128) / 256.0 + 1
    return (cal_x, cal_y, cal_z)

def read_mpu_and_mag(b, mag_cal):
    bytes_ = bytes(b.read_i2c_block_data(MPU_ADDR, 0x3B, 14 + 8))
    mpu_bytes = bytes_[:14]
    mag_bytes = bytes_[14:]
    mpu_data = struct.unpack(">hhhhhhh", mpu_bytes)
    mag_data = struct.unpack("<chhhc", mag_bytes)

    accel_data = mpu_data[0:3]
    temp_sens = mpu_data[3]
    gyro_data = mpu_data[4:8]
    mag_data_real = [x * y for x, y in zip(mag_data[1:4], mag_cal)]

    return (accel_data, gyro_data, mag_data_real, temp_sens)

def demo():
    b = smbus.SMBus(6)
    setup_mpu(b)
    mag_cal = setup_mag(b)
    print("Magnetometer cal: {}".format(mag_cal))
    while True:
        print(read_mpu_and_mag(b, mag_cal))

def demo2():
    os.nice(-20)
    dT = 0.05
    x = filterSetup()
    x[0].deltat = dT
    b = smbus.SMBus(6)
    setup_mpu(b)
    mag_cal = setup_mag(b)
    print("Magnetometer cal: {}".format(mag_cal))

    gyro_rads_per_s_per_lsb = 250.0 / 180.0 * math.pi / 32768.0

    while True:
        time_now = time.time()
        accel_data, gyro_data, mag_data, _ = read_mpu_and_mag(b, mag_cal)
        filterUpdate(x,
            gyro_data[0] * gyro_rads_per_s_per_lsb,
            gyro_data[1] * gyro_rads_per_s_per_lsb,
            gyro_data[2] * gyro_rads_per_s_per_lsb,
            accel_data[0], accel_data[1], accel_data[2],
            mag_data[1], mag_data[0], mag_data[2])
        print("{} {} {} {}".format(x[0].SEq_1, x[0].SEq_2, x[0].SEq_3, x[0].SEq_4))
        # print("Exec took {} ms".format((time.time() - time_now) * 1000))
        time.sleep(dT - (time.time() - time_now))

def demo3():
    IP = '192.168.0.2'
    PORT = 5005

    os.nice(-20)
    dT = 0.05
    x = filterSetup()
    x[0].beta = math.sqrt(3/4) * math.pi * 40 / 180
    x[0].zeta = 0
    x[0].deltat = dT
    b = smbus.SMBus(6)
    setup_mpu(b)
    mag_cal = setup_mag(b)
    print("Magnetometer cal: {}".format(mag_cal))

    gyro_rads_per_s_per_lsb = 250.0 / 180.0 * math.pi / 32768.0

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_sock.connect((IP, PORT))

    while True:
        time_now = time.time()
        accel_data, gyro_data, mag_data, _ = read_mpu_and_mag(b, mag_cal)
        filterUpdate(x,
            gyro_data[0] * gyro_rads_per_s_per_lsb,
            gyro_data[1] * gyro_rads_per_s_per_lsb,
            gyro_data[2] * gyro_rads_per_s_per_lsb,
            accel_data[0], accel_data[1], accel_data[2],
#             mag_data[1], mag_data[0], mag_data[2])
            1, 0, 0)
        # print("{} {} {} {}".format(x[0].SEq_1, x[0].SEq_2, x[0].SEq_3, x[0].SEq_4))

        bytes_ = struct.pack('ffff', x[0].SEq_1, x[0].SEq_2, x[0].SEq_3, x[0].SEq_4)
        udp_sock.send(bytes_)
        # print("Exec took {} ms".format((time.time() - time_now) * 1000))

        time.sleep(dT - (time.time() - time_now))

if __name__=='__main__':
    demo3()
