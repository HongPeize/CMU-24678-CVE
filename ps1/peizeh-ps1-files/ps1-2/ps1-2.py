import cv2
import numpy as np

def binary2R(img, BorD):
    cv2.imshow('img', img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # convert to grayscale
    cv2.imshow('img_grayscale.png', grayscale)

    (thresh, binary) = cv2.threshold(grayscale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #convert to binary
    cv2.imshow('img_binary.png', binary)

    binary3c = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB) #convert binary image to 3 channel
    if BorD == "b":
        binary3c[binary == 255] = (0,0,255)
    elif BorD == "d":
        binary3c[binary == 0] = (0, 0, 255)
    cv2.imshow('img_ouput.png', binary3c)

path = "circuit.png"
image = cv2.imread(path)

BrightorDark = str(input("Turn bright or dark part? Type [b] or [d] "))
imgsv = str(input("Do you want to save the images? Type [y] or [n] "))

binary2R(image, BrightorDark)
if imgsv == "y":
    cv2.imwrite('img_grayscale.png', grayscale)
    cv2.imwrite('img_binary.png', binary)
    cv2.imwrite('img_ouput.png', binary3c)
cv2.waitKey(0)
