import sys
import math
import cv2 as cv
import numpy as np

def main(argv):
    
    default_file = 'app/img.jpg'
    # default_file = 'app/all_border_dataset_2000/Img_000001.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)
    
    # vertical lines
    vert_lines = cv.HoughLines(dst, rho=1, theta=np.pi/90, threshold=200, min_theta=0, max_theta=np.pi/45)

    # horizontal lines
    hor_lines = cv.HoughLines(dst, rho=1, theta=np.pi/90, threshold=200, min_theta=np.pi/2-np.pi/90, max_theta=np.pi/2+np.pi/90)
    
    # print(vert_lines.shape)
    # print(hor_lines.shape)

    lines = np.concatenate((vert_lines, hor_lines))

    if lines is not None:
        print(len(lines))
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
    
    
    # linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv.LINE_AA)
    
    cv.imwrite("Source.jpg", src)
    cv.imwrite("Detected Lines (in red) - Standard Hough Line Transform.jpg", cdst)
    # cv.imwrite("Detected Lines (in red) - Probabilistic Line Transform.jpg", cdstP)
    
if __name__ == "__main__":
    main(sys.argv[1:])