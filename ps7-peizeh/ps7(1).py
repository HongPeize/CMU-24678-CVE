import cv2
import numpy as np

def depth_map(disp, frame_name, img_l):
        h, w = img_l.shape[:2]
        f = 0.8 * w  # guess the focal length
        Q = np.float32([[1, 0, 0, -0.5 * w],
                        [0, -1, 0, 0.5 * h],
                        [0, 0, 0, -3 * f],
                        [0, 0, 1, 100]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(img_l, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        points = points[mask].reshape(-1, 3)
        colors = colors[mask].reshape(-1, 3)

        points = np.hstack([points, colors])
        with open(frame_name, 'wb') as f:
                f.write((ply_header % dict(vert_num=len(points))).encode('utf-8'))
                np.savetxt(f, points, fmt='%f %f %f %d %d %d ')


img_l = cv2.imread("ball-left.png")
img_r = cv2.imread("ball-right.png")

min_disp = 5
num_disp = 3*16

#min_disp = 8
#num_disp = 4*16

window_size = 8
left_matcher = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=2*window_size,
        P1=8 * 3 * window_size**2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=1,
        mode=cv2.StereoSGBM_MODE_SGBM_3WAY
    )

ply_header = '''ply
format ascii 1.0
comment - 24678 PS7
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

disp = left_matcher.compute(img_l, img_r).astype(np.float32) / 16.0
output_img = (disp-min_disp)/num_disp

# generate 3d point cloud
frame_name = 'ball.ply'
depth_map(disp, frame_name, img_l)

cv2.imshow("img",255*(output_img))
cv2.waitKey(0)