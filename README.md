# Various methods for checking of borders in images
app/ uses ENIMDA algorithm, also has dataset splitting notebook for CNN due to location of dataset (TODO: move it)

cnn/ uses faster-rcnn resnet50 fpn

hough/ uses Canny + Hough

## Results:
#### ENIMDA
TP: 172, TN: 580, FP: 1222, FN: 28
86% TP 32.19% TN

#### CNN
TP: 18, TN:157, FP: 25, FN: 1
94.74% TP, 86.26% TN

#### Canny + Hough
TP: 153, TN: 1344, FP: 458, FN: 47
76.5% TP, 74.54% TN
