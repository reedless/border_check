import os
import cv2
import numpy as np

# img = cv2.imread("app/img.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# canny  = cv2.Canny(blurred, 75, 150)

# # Find contours
# cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# # Iterate thorugh contours and draw rectangles around contours
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (36,255,12), 2)

# cv2.imwrite('canny.jpg', canny)
# cv2.imwrite('image.jpg', img)

# img = cv2.imread("app/img.jpg", -1)

img = cv2.imread("app/img.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (3, 3), 0)
canny  = cv2.Canny(blurred, 75, 150)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# binary = cv2.bitwise_not(canny)

contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = contours[0] if len(contours) == 2 else contours[1]

for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

cv2.imwrite('image.jpg', img)


# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30, maxLineGap=250)
# cv2.imwrite('res.jpg', lines)

# TP = 0
# TN = 0
# FP = 0
# FN = 0

# test_dir= 'app/all_border_dataset_2000'

# for i in sorted(os.listdir(test_dir)):
#     if i[-15:] == 'Zone.Identifier':
#         continue
#     image = cv2.imread(f"{test_dir}/{i}")
#     res = Check(image)
#     if i[:6] == 'border':
#         if not res.check_border():
#             print(f"{i} is incorrect")
#             FN += 1
#         else:
#             TP += 1
#     else:
#         if res.check_border():
#             print(f"{i} is incorrect")
#             FP += 1
#         else:
#             TN += 1

# print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")