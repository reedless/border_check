import math
import cv2 as cv
import numpy as np
import os

def main():
    loop(130, 20)
    # best = 0
    # res = (0, 0, 0, 0)
    # for threshold in range(50, 141, 5):
    #     for max_lines in range(10, 51, 5):
    #         print(threshold, max_lines)
    #         TP, TN, _, _ = loop(threshold, max_lines)
    #         metric = TP*18 + TN
    #         if metric > best:
    #             best = metric
    #             print(TP, TN)
    #             res = (TP, TN, threshold, max_lines)

    # TP, TN, threshold, max_lines = res
    # print(f"threshold={threshold}, max_lines={max_lines}")
    # print(f"TP: {TP}, TN: {TN}, FP: {1802-TN}, FN: {200-TP}")

def loop(threshold, max_lines):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # test_dir= 'app/curated_border_dataset_cache'
    test_dir= '../app/all_border_dataset_2000'

    for i in sorted(os.listdir(test_dir)):
        if i[-15:] == 'Zone.Identifier':
            continue
        # if i ends with _lines.jpg
        if i[-10:] == '_lines.jpg':
            continue
        filename = f"{test_dir}/{i}"
        lines, cdst = check_border(filename, threshold)

        if i[:6] == 'border':
            if lines is None or len(lines) > max_lines:
                # print(f"{i} is incorrect")
                # if lines is not None:
                #     print(len(lines))
                FN += 1
            else:
                # print(len(lines))
                TP += 1
        else:
            if lines is not None and len(lines) <= max_lines:
                # print(f"{i} is incorrect")
                # print(len(lines))
                FP += 1
            else:
                # if lines is not None:
                #     print(len(lines))
                TN += 1

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    return TP, TN, FP, FN
    
def check_border(filename, threshold=100):

    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    dst = cv.Canny(src, 50, 200, None, 3)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # vertical lines
    vert_lines_1 = cv.HoughLines(dst, rho=1, theta=np.pi/180, threshold=threshold, min_theta=0, max_theta=np.pi/180*5)
    vert_lines_2 = cv.HoughLines(dst, rho=1, theta=np.pi/180, threshold=threshold, min_theta=np.pi/180*175, max_theta=np.pi)
    if vert_lines_1 is not None:
        if vert_lines_2 is not None:
            vert_lines = np.concatenate((vert_lines_1, vert_lines_2), axis=0)
        else:
            vert_lines = vert_lines_1
    elif vert_lines_2 is not None:
        vert_lines = vert_lines_2
    else:
        vert_lines = None

    # horizontal lines
    hor_lines = cv.HoughLines(dst, rho=1, theta=np.pi/180, threshold=threshold, min_theta=np.pi/180*85, max_theta=np.pi/180*95)

    # concat vertical and horizxonal lines if they are not none
    if vert_lines is not None:
        if hor_lines is not None:
            lines = np.concatenate((vert_lines, hor_lines))
        else:
            lines = vert_lines
    elif hor_lines is not None:
        lines = hor_lines
    else:
        lines = None

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return lines, cdst
    
    
    
if __name__ == "__main__":
    main()