#!/usr/bin/env python3

from ctypes import *

class filt_state(Structure):
    _fields_ = [('deltat', c_float),
                ('beta', c_float),
                ('zeta', c_float),
                ('SEq_1', c_float),
                ('SEq_2', c_float),
                ('SEq_3', c_float),
                ('SEq_4', c_float),
                ('SEqDot_1', c_float),
                ('SEqDot_2', c_float),
                ('SEqDot_3', c_float),
                ('SEqDot_4', c_float),
                ('b_x', c_float),
                ('b_z', c_float),
                ('w_bx', c_float),
                ('w_by', c_float),
                ('w_bz', c_float),]

__thelib = cdll.LoadLibrary('./Madgwick_imu.so')

filterSetup = __thelib.filterSetup
filterSetup.restype = POINTER(filt_state)

filterUpdate = __thelib.filterUpdate
filterUpdate.argtypes = [POINTER(filt_state), c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float, c_float]
