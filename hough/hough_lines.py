import sys
import math
import cv2 as cv
import numpy as np

def main(argv):
    
    default_file = '../app/img.jpg'
    # default_file = '../app/all_border_dataset_2000/Img_000001.jpg'
    filename = argv[0] if len(argv) > 0 else default_file
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image!')
        print ('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    
    
    # dst = cv.Canny(src, 50, 200, None, 3)
    
    # Copy edges to the images that will display the results in BGR
    cdst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
    # cdstP = np.copy(cdst)
    
    # vertical lines
    vert_lines = cv.HoughLines(src, rho=1, theta=np.pi/90, threshold=200, min_theta=0, max_theta=np.pi/45)

    # horizontal lines
    hor_lines = cv.HoughLines(src, rho=1, theta=np.pi/90, threshold=200, min_theta=np.pi/2-np.pi/90, max_theta=np.pi/2+np.pi/90)
    
    print(vert_lines.shape)
    print(hor_lines.shape)

    strong_lines_vert = np.zeros([4,1,2])
    strong_lines_hor = np.zeros([4,1,2])

    j = 0
    for i in range(len(vert_lines)):
        for rho,theta in vert_lines[i]:
            if i == 0:
                strong_lines_vert[j] = vert_lines[i]
                j += 1
            else:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                closeness_rho = np.isclose(rho, strong_lines_vert[0:j,0,0],atol = 10)
                closeness_theta = np.isclose(theta,strong_lines_vert[0:j,0,1],atol = np.pi/36)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and j < 4:
                    strong_lines_vert[j] = vert_lines[i]
                    j += 1

    j = 0
    for i in range(len(hor_lines)):
        for rho,theta in hor_lines[i]:
            if i == 0:
                strong_lines_hor[j] = hor_lines[i]
                j += 1
            else:
                if rho < 0:
                    rho*=-1
                    theta-=np.pi
                closeness_rho = np.isclose(rho, strong_lines_hor[0:j,0,0],atol = 10)
                closeness_theta = np.isclose(theta,strong_lines_hor[0:j,0,1],atol = np.pi/36)
                closeness = np.all([closeness_rho,closeness_theta],axis=0)
                if not any(closeness) and j < 4:
                    strong_lines_hor[j] = hor_lines[i]
                    j += 1

    lines = np.concatenate((strong_lines_vert, strong_lines_hor))

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