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
    test_dir= '../app/all_border_dataset_2000'

    for i in sorted(os.listdir(test_dir)):
        if i[-15:] == 'Zone.Identifier':
            continue
        # if i ends with _lines.jpg
        if i[-10:] == '_lines.jpg':
            continue
        filename = f"{test_dir}/{i}"
        lines, cdst = check_border(filename)
        max_lines = 10

        if i[:6] == 'border':
            if lines is None or len(lines) > max_lines:
                print(f"{i} is incorrect")
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                FN += 1
            else:
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                print(len(lines))
                TP += 1
        else:
            if lines is not None and len(lines) <= max_lines:
                print(f"{i} is incorrect")
                print(len(lines))
                cv.imwrite(f"{test_dir}/{i}_lines.jpg", cdst)
                FP += 1
            else:
                TN += 1

    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    
def check_border(filename):

    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    height, width = src.shape
    dst = cv.Canny(src, 50, 200, None, 3)
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    linesP = cv.HoughLinesP(dst, rho=1, theta=np.pi/90, threshold=30, minLineLength=0.5*min([height, width]), maxLineGap=5)
    
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)

    return linesP, cdstP
    
    
    
if __name__ == "__main__":
    main()