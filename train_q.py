#! /usr/bin/env python
"""
Execute a training run of deep-Q-Leaning.

"""

import sys
import launcher

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000 # 250000
    EPOCHS = 8000
    STEPS_PER_TEST = 130000 # 130000

    # ----------------------
    # ALE Parameters
    # ----------------------
    BASE_ROM_PATH = "../aleroms/"
    ROM = 'bug.bin'
    FRAME_SKIP = 4

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 #0.95 (Rho)
    RMS_EPSILON = .01 #0.01
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1 #or 0.01 for tuned ddqn
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 10000 #30000 for tuned ddqn
    REPLAY_START_SIZE = 50000 #50000
    RESIZE_METHOD = 'scale' #scale vs crop
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    OFFSET = 18
    DEATH_ENDS_EPISODE = True
    CAP_REWARD = True
    MAX_START_NULLOPS = 30
    OPTIMAL_EPS = 0.05 #0.05 or 0.001 for tuned ddqn
    DOUBLE_Q = False
    MEAN_FRAME = False
    TEMP=1

    TERMINATION_REG = 0.0
    NUM_OPTIONS = 8
    ACTOR_LR = 0.00025
    ENTROPY_REG = 0.0
    BASELINE=False

if __name__ == "__main__":
    launcher.launch(sys.argv[1:], Defaults, __doc__)
