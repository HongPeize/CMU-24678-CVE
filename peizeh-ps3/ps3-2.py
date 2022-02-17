import cv2
import numpy as np

path = "cheerios.png"
img = cv2.imread(path)
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ksize = 3

#vertical deravitive
A = np.zeros((int((ksize + 1) / 2), 1))
for i in range((ksize + 1) // 2):
    A[i, 0] = i + 1
A = np.vstack((A, A[:ksize // 2][::-1]))
B = np.zeros((1,ksize))
for j in range (ksize):
    B[0,j] = ksize//2 - j
ver_filter = A * B

#horizontal deravitive
hor_filter = np.array([[1.,2.,1.],
                        [0,0,0],
                        [-1.,-2.,-1.]])

img_row, img_col = np.shape(grayscale)
new_img = np.zeros((img_row, img_col))


for i in range(ksize//2, img_row - ksize//2):
    for j in range(ksize//2, img_col - ksize//2):
        sub_area = grayscale[i - ksize//2 : i + ksize//2 + 1, j - ksize//2 : j + ksize//2 + 1]
        new_img[i,j] = np.sqrt( (np.sum(np.multiply(ver_filter, sub_area)))**2 + (np.sum(np.multiply(hor_filter, sub_area)))**2 )

new_img = np.array(new_img, np.uint8)
(thresh, img_binary) = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
invert = cv2.bitwise_not(img_binary)
cv2.imshow("img_output", invert)

#Canny Filter with Trackbar
def Canny(_):
    Lower_Threshold = cv2.getTrackbarPos('Min Threshold:', 'Canny Filter')
    Upper_Threshold = cv2.getTrackbarPos('Max Threshold:', 'Canny Filter')
    Kernel_size = cv2.getTrackbarPos('Kernel Size:', 'Canny Filter')
    L2_grad = cv2.getTrackbarPos('L2 Gradient:', 'Canny Filter')
    if Kernel_size == 0:
        k_size = 3
    elif Kernel_size == 1:
        k_size = 5
    elif Kernel_size == 2:
        k_size = 7
    detected_edges = cv2.Canny(grayscale, Lower_Threshold, Upper_Threshold, apertureSize=k_size, L2gradient=L2_grad)
    canny_invert = cv2.bitwise_not(detected_edges)
    font = cv2.FONT_HERSHEY_SIMPLEX
    if path =="circuit.png":
        canny_invert = cv2.putText(canny_invert, "Lower Threshold: "+ str(Lower_Threshold), (1050, 890), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Upper Threshold: "+ str(Upper_Threshold), (1050, 905), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Kernel Size: " + str(k_size), (1050, 920), font, 0.5,
                               (0, 0, 255), 1)
    elif path =="cheerios.png":
        canny_invert = cv2.putText(canny_invert, "Lower Threshold: "+ str(Lower_Threshold), (844, 728), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Upper Threshold: "+ str(Upper_Threshold), (844, 743), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Kernel Size: " + str(k_size), (844, 758), font, 0.5,
                               (0, 0, 255), 1)
    elif path =="gear.png":
        canny_invert = cv2.putText(canny_invert, "Lower Threshold: "+ str(Lower_Threshold), (483, 489), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Upper Threshold: "+ str(Upper_Threshold), (483, 504), font, 0.5, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Kernel Size: " + str(k_size), (483, 519), font, 0.5,
                               (0, 0, 255), 1)
    elif path =="professor.png":
        canny_invert = cv2.putText(canny_invert, "Lower Threshold: "+ str(Lower_Threshold), (1500, 42), font, 1, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Upper Threshold: "+ str(Upper_Threshold), (1500, 72), font, 1, (0, 0, 255), 1)
        canny_invert = cv2.putText(canny_invert, "Kernel Size: " + str(k_size), (1500, 102), font, 1,
                               (0, 0, 255), 1)

    cv2.imshow("Canny", canny_invert)
    cv2.imwrite(path+"-canny.png", canny_invert)

min_lowThreshold = 0
max_lowThreshold = 200
max_highThreshold = 300

cv2.namedWindow('Canny Filter')
cv2.createTrackbar('Kernel Size:', 'Canny Filter', 0, 2, Canny)
cv2.createTrackbar('Min Threshold:', 'Canny Filter', min_lowThreshold, max_lowThreshold, Canny)
cv2.createTrackbar('Max Threshold:', 'Canny Filter', max_lowThreshold+1, max_highThreshold, Canny)
cv2.createTrackbar('L2 Gradient:', 'Canny Filter', False, True, Canny)

Canny(0)
cv2.waitKey()
#cv2.imwrite(path+"-sobel.png",invert)



