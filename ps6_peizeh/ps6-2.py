import cv2
import numpy as np

img = cv2.imread("spade-terminal.png")
template_img = cv2.imread("template.png")


def defective_contour(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr, dst = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
    for i in range(1):
        dst = cv2.erode(dst, None)
    for i in range(1):
        dst = cv2.dilate(dst, None)
    cont, hier = cv2.findContours(cv2.bitwise_not(dst), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def complete_contour(img):  # template
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to gray-scale
    thr, dst = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)  # Binary
    for i in range(1):
        dst = cv2.erode(dst, None)
    for i in range(1):
        dst = cv2.dilate(dst, None)
    cont, hier = cv2.findContours(cv2.bitwise_not(dst), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return cont


spade_contour = complete_contour(template_img)  # 1 ref contour
def_contour = defective_contour(img)
def_list = []
for i in def_contour:
    diff = cv2.matchShapes(i, spade_contour[0], cv2.CONTOURS_MATCH_I2, 0)
    if diff > 2:
        def_list.append(i)
img = cv2.drawContours(img, def_list, -1, (0, 0, 255), -1)
cv2.imwrite("spade-terminal-output.png", img)
cv2.imshow("spade-terminal-output", img)
cv2.waitKey(0)