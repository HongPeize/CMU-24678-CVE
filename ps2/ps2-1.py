import cv2
import numpy as np

def pseudo(img, s):
    r_table = []
    g_table = []
    b_table = []
    x_coord = 0
    y_coord = 0
    index_list = []

    for i in range(256):
        if i < 85:
            r_table.append(0)
        elif i>84 and i<171:
            r_table.append(3 * i - 255)
        else:
            r_table.append(255)
    r_table = np.array(r_table, np.uint8)

    for j in range(256):
        if j < 85:
            g_table.append(3 * j)
        elif j>84 and j<171:
            g_table.append(255)
        else:
            g_table.append(-3 * j + 765)
    g_table = np.array(g_table, np.uint8)

    for k in range(256):
        if k < 85:
            b_table.append(255)
        elif k>84 and k<171:
            b_table.append(-3 * k + 510)
        else:
            b_table.append(0)
    b_table = np.array(b_table, np.uint8)

    lut = np.dstack((b_table, g_table, r_table))

    cv2.imshow('img.png', img)
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3c = cv2.merge([grayscale, grayscale, grayscale])
    min_int = np.min(grayscale)  # find min pixel intensity value in img
    max_int = np.max(grayscale)  # find max pixel intensity value in img
    img_shape = list(np.shape(grayscale))
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if grayscale[i,j] == max_int:
                index_list.append([j,i]) # return a list of indices with shared max value
    for x,y in index_list:
        x_coord += x
        y_coord += y
    cog = (x_coord/len(index_list), y_coord/len(index_list))
    pseudo_img = cv2.LUT(gray3c, lut)
    cv2.line(pseudo_img, (int(cog[0])-20, int(cog[1])), (int(cog[0])+20, int(cog[1])), (255, 255, 255), 2)
    cv2.line(pseudo_img, (int(cog[0]), int(cog[1]) - 20), (int(cog[0]), int(cog[1]) + 20), (255, 255, 255), 2)
    cv2.circle(pseudo_img, (int(cog[0]), int(cog[1])), 15, (255, 255, 255), 2)  # place cross at CoG
    cv2.imshow('img-color.png', pseudo_img)
    if s == "y":
        cv2.imwrite('img-color.png', pseudo_img)

switch = input("Save file? [y][n]")
path = "thermal.png"
image = cv2.imread(path)
pseudo(image, switch)
cv2.waitKey(0)