import sys
import math
import cv2 as cv
import numpy as np
import os
from app.bccaas_engine.upload import UploadFile


def main():

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # test_dir= 'app/curated_border_dataset_cache'
    test_dir= 'app/all_border_dataset_2000'

    for i in sorted(os.listdir(test_dir)):
        if i[-15:] == 'Zone.Identifier':
            continue
        # if i ends with _lines.jpg
        if i[-10:] == '_lines.jpg':
            continue
        filename = f"{test_dir}/{i}"
        lines, cdst = check_border(filename)

        if i[:6] == 'border':
            if lines is None:
                print(f"{i} is incorrect")
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                FN += 1
            else:
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                print(len(lines))
                TP += 1
        else:
            if lines is not None and len(lines) < 100: # 100 is arbitrary, can adjust
                print(f"{i} is incorrect")
                print(len(lines))
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                FP += 1
            else:
                TN += 1

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    
def check_border(filename):

    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    dst = cv.Canny(src, 50, 200, None, 3)
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    # vertical lines
    vert_lines = cv.HoughLines(dst, rho=1, theta=np.pi/90, threshold=150, min_theta=0, max_theta=np.pi/45)

    # horizontal lines
    hor_lines = cv.HoughLines(dst, rho=1, theta=np.pi/90, threshold=150, min_theta=np.pi/2-np.pi/90, max_theta=np.pi/2+np.pi/90)

    lines = np.concatenate((vert_lines, hor_lines))l

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

    return lines, cdst,
    
    
    
if __name__ == "__main__":
    main()