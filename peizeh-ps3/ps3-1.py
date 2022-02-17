import cv2
import numpy as np

path = "golf.png"
img = cv2.imread(path)

kernel = np.array([
  [0., -1., 0.],
  [-1., 5., -1.],
  [0., -1., 0.]
]) #sharpen filter
if path == "pcb.png":
  img_blur = cv2.medianBlur(img, 3)
  sharpen_img = cv2.filter2D(img_blur, -1, kernel)

elif path == "pots.png":
  img_blur = cv2.GaussianBlur(img, (11,11), 6)
  sharpen_img = cv2.addWeighted(img, 2.5, img_blur, -1.6, 0)
elif path == "rainbow.png":
  img_blur = cv2.bilateralFilter(img, 12, 60, 60)
  sharpen_img = cv2.filter2D(img_blur, -1, kernel)
elif path =="golf.png":
  img_blur = cv2.medianBlur(img, 5)
  sharpen_img = cv2.filter2D(img_blur, -1, kernel)

cv2.imshow("sharpen", sharpen_img)
cv2.imwrite(path+"-improved.png", sharpen_img)
cv2.waitKey(0)