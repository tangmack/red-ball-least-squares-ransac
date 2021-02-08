#!/usr/bin/python3

# ========================================
# ENPM673 Spring 2021: Perception for Autonomous Robotics
# Example script for getting started with OpenCV
#
# Author: Mack Tang (macktang@gmail.com)
# ========================================
# Run as 'python3 process_video.py'
# Press q for exit


import argparse
import numpy as np
import cv2
import math


def main():
    cap = cv2.VideoCapture('Ball_travel_10fps.mp4')
    # cap = cv2.VideoCapture('Ball_travel_2_updated.mp4')

    row_list = []
    col_list = []

    frame_count = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        # print(type(frame))
        # print(frame.shape)
        # print(frame)

        green_indices = np.where(frame[:, :, 1] < 200)  # where green is less than 200

        # print(green_indices)

        green_rows = green_indices[0]
        green_cols = green_indices[1]

        highest_row_index = np.argmin(green_rows)
        lowest_row_index = np.argmax(green_rows)
        # Now that we have the index of the highest row and lowest row (wrt the collection of green-only pixels)
        # we must get the actual highest row and highest row
        highest_row = green_rows[highest_row_index]
        lowest_row = green_rows[lowest_row_index]

        leftmost_col_index = np.argmin(green_cols)
        rightmost_col_index = np.argmax(green_cols)
        # Now that we have the index of the leftmost col and rightmost col (wrt the collection of green-only pixels)
        # we must get the actual leftmost col and rightmost col
        leftmost_col = green_cols[leftmost_col_index]
        rightmost_col = green_cols[rightmost_col_index]
        # Finally, get average "center" column
        center_col = round((leftmost_col + rightmost_col)/2)
        print(center_col)



        # high_center_col = np.argmin(frame[highest_row,:,1])
        # low_center_col = np.argmin(frame[lowest_row,:,1])


        # highest_col = green_cols[highest_row_index]
        # lowest_col = green_cols[lowest_row_index]


        frame[highest_row,center_col,2] = 0
        frame[lowest_row,center_col,2] = 0

        # frame[highest_row_index,highest_col_index,:] = np.array([200,100,0])
        # frame[lowest_row_index,lowest_col_index,:] = np.array([200,100,0])

        # green_y = np.abs(green_rows - frame.shape[0])
        # green_x = np.abs(green_cols - frame.shape[1])





        # cv2.imshow('frame', frame[:, :, 1])
        cv2.imshow('frame', frame)
        k = cv2.waitKey(0) & 0xFF
        if k == 115:  # if s key pressed
            cv2.imwrite('frame' + str(frame_count) + '.png', frame)
            print("saved!")
        elif k == ord('q'):
            break

        frame_count += 1
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
