import numpy as np
import cv2

def GammaCorrection(image, gamma):
	# build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
	invGamma = 1.0 / gamma
	table = [((i / 255.0) ** invGamma) * 255 for i in np.arange(256)] #i: output pixel intensity
	table = np.array(table, np.uint8) #map between corrected and uncorrected pixel

	return cv2.LUT(image, table) # apply gamma correction using the lookup table

path = "carnival.jpg"
img = cv2.imread(path)
g = float(input("Gamma value: "))
imgsv = str(input("Do you want to save the images? Type [y] or [n] "))

corrected = GammaCorrection(img, g) #apply gamma correction

cv2.putText(img, "Orignal", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 3) #Put text on image
cv2.imshow("img", img) #show image

cv2.putText(corrected, "g={}".format(g), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 3) #Put text on image
cv2.imshow("img_gcorrected", corrected) #show image
if imgsv == "y":
	cv2.imwrite('img_corrected.png', corrected)

cv2.waitKey(0)