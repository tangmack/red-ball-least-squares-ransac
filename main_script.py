#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
#
#
# Author: Mack Tang (macktang@gmail.com)
# ========================================
# See Readme.md for instructions to run

from process_video import main
from least_squares import f2
from total_least_squares import f3
from RANSAC import f4

import matplotlib.pyplot as plt

if __name__ == '__main__':
    fig = plt.figure()
    plt.axis([0, 10, 0, 10])
    t = ("Keep pressing q key to continue, and see all graphs. "
         "First, graphs with no noise... ")
    plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',
             va='top', wrap=True)
    plt.show()

    main(0) # 0 for no noise
    f2()
    f3()
    f4()

    # Now do noisy graphs
    fig = plt.figure()
    plt.axis([0, 10, 0, 10])
    t = ("Noisy graphs up next...")
    plt.text(5, 10, t, fontsize=18, style='oblique', ha='center',
             va='top', wrap=True)
    plt.show()

    main(1) # 1 for noisy
    f2()
    f3()
    f4()