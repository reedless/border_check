import math
import cv2 as cv
import numpy as np
import os

def main():
    # for i in range(130, 201, 10):
    #     print(i)
    #     loop(i, 5000, 5000)
    best = 0
    res = (0, 0, 0, 0, 0)
    for threshold in range(120, 151, 10):
        for max_vert_lines in range(5, 21, 5):
            for max_hor_lines in range(5, 21, 5):
                print(f"threshold: {threshold}, max_vert_lines: {max_vert_lines}, max_hor_lines: {max_hor_lines}")
                TP, TN, FP, FN = loop(threshold, max_vert_lines, max_hor_lines)
                metric = TP*9 + TN
                if metric > best:
                    best = metric
                    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
                    res = ( TP, TN, threshold, max_vert_lines, max_hor_lines)

    TP, TN, threshold, max_vert_lines, max_hor_lines = res
    print(f"threshold={threshold}, max_vert_lines={max_vert_lines}, max_hor_lines={max_hor_lines}")
    print(f"TP: {TP}, TN: {TN}, FP: {1802-TN}, FN: {200-TP}")

    # best = 0
    # res = (0, 0, 0, 0)
    # for threshold in range(50, 141, 10):
    #     for max_lines in range(10, 51, 10):
    #         print(threshold, max_lines)
    #         TP, TN, _, _ = loop(threshold, max_lines)
    #         metric = TP*9 + TN
    #         if metric > best:
    #             best = metric
    #             print(TP, TN)
    #             res = (TP, TN, threshold, max_lines)

    # TP, TN, threshold, max_lines = res
    # print(f"threshold={threshold}, max_lines={max_lines}")
    # print(f"TP: {TP}, TN: {TN}, FP: {1802-TN}, FN: {200-TP}")

def loop(threshold, max_vert_lines, max_hor_lines):

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
        vert_lines, hor_lines, cdst = check_border(filename, threshold)

        if i[:6] == 'border':
            if not has_border(vert_lines, hor_lines, max_vert_lines, max_hor_lines, cdst):
                # print(f"{i} is incorrect")
                # print_len_lines(vert_lines, hor_lines)
                FN += 1
            else:
                # print(f"{i} is correct")
                # print_len_lines(vert_lines, hor_lines)
                TP += 1
        else:
            if has_border(vert_lines, hor_lines, max_vert_lines, max_hor_lines, cdst):
                # print(f"{i} is incorrect")
                # print_len_lines(vert_lines, hor_lines)
                FP += 1
            else:
                TN += 1

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    return TP, TN, FP, FN

def has_lines_in_center(dst, lines):
    for i in range(len(lines)):
        # add to new_lines if line is at edge
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        (x1, y1), (x2, y2) = compute_line_endpoints(rho, theta)
        height, width, _ = dst.shape

        if not not_in_center_of_image(x1, y1, x2, y2, width, height):
            return True

    return False

def has_border(vert_lines, hor_lines, max_vert_lines, max_hor_lines, cdst):
    # no lines found
    if vert_lines is None and hor_lines is None:
        return False

    # only hor lines found
    if vert_lines is None:
        if has_lines_in_center(cdst, hor_lines):
            return False
        return len(hor_lines) < max_vert_lines

    # only vert lines found
    if hor_lines is None:
        if has_lines_in_center(cdst, vert_lines):
            return False
        return len(vert_lines) < max_hor_lines

    # both lines found
    if has_lines_in_center(cdst, vert_lines) or has_lines_in_center(cdst, hor_lines):
        return False
    return len(vert_lines) < max_vert_lines and len(hor_lines) < max_hor_lines

def print_len_lines(vert_lines, hor_lines):
    if vert_lines is not None:
        print(len(vert_lines))
    if hor_lines is not None:
        print(len(hor_lines))
    
def check_border(filename, threshold=100):

    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    high_thresh, _ = cv.threshold(src, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    lowThresh = 0.5*high_thresh
    dst = cv.Canny(src, lowThresh, high_thresh, None, 3)
    # dst = cv.Canny(src, 50, 200, None, 3)
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

    # vert_lines = filter_only_lines_at_edges(dst, vert_lines)
    
    # horizontal lines
    hor_lines = cv.HoughLines(dst, rho=1, theta=np.pi/180, threshold=threshold, min_theta=np.pi/180*85, max_theta=np.pi/180*95)

    # hor_lines = filter_only_lines_at_edges(dst, hor_lines)

    # concat vertical and horizonal lines if they are not none
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
        for i in range(len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            pt1, pt2 = compute_line_endpoints(rho, theta)
            cv.line(cdst, pt1, pt2, (0,0,255), 3, cv.LINE_AA)

    return vert_lines, hor_lines, cdst

def filter_only_lines_at_edges(dst, lines):
    if lines is not None:
        new_lines = np.zeros(lines.shape)
        j = 0
        for i in range(len(lines)):
            # add to new_lines if line is at edge
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            (x1, y1), (x2, y2) = compute_line_endpoints(rho, theta)
            height, width = dst.shape

            if not_in_center_of_image(x1, y1, x2, y2, width, height):
                new_lines[j] = lines[i]
                j += 1

        return new_lines[:j,:,:]

    else:
        return lines

def not_in_center_of_image(x1, y1, x2, y2, width, height, factor=0.2):
    return (x1 < width*factor or x1 > width*(1-factor)) and (x2 < width*factor or x2 > width*(1-factor)) and \
            (y1 < height*factor or y1 > height*(1-factor)) and (y2 < height*factor or y2 > height*(1-factor))

def compute_line_endpoints(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    return (x1, y1), (x2, y2)
    
    
    
if __name__ == "__main__":
    main()