#!/usr/bin/env python3

import ev3dev.ev3 as ev3

mot_rear = ev3.LargeMotor('outA')
mot_rear.run_forever(speed_sp=200)
mot_kick = ev3.LargeMotor('outD')
# Pushing
# mot_kick.run_timed(time_sp=3000, speed_sp=500)
# Retracting
# mot_kick.run_timed(time_sp=3000, speed_sp=-500)
mot_steer = ev3.MediumMotor('outB')
# Towards left
# mot_steer.run_timed(time_sp=3000, speed_sp=500)
