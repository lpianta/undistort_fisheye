# Import libraries

import cv2
import numpy as np

# Constants

# X, Y of pic
DIM = (3040, 3040)
# Camera Matrix
K = np.array([[1.26073776e+03, 0.00000000e+00, 1.47237387e+03],
              [0.00000000e+00, 1.48115434e+03, 1.74279556e+03],
              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# Distortion
D = np.array(
    [[-0.33131782,  0.07956073, -0.00959671,  0.00071889, -0.00816541]])


# Functions

# Cut the original img in 2 halves
def cut(img_path):
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    right_img = img[0:height, 0:width//2]
    left_img = img[0:height, width//2:width]
    return left_img, right_img


def undistort(img, K=K, D=D, dim=DIM):
    map1, map2 = cv2.initUndistortRectifyMap(
        K, D, np.eye(3), K, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(
        img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def rotate(img, rot):
    if rot == "counterclock":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot == "clock":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)


def final(left, right):
    return np.concatenate((left, right), axis=1)


left, right = cut("360img/IMG_20210621_103023_00_024.jpg")
und_left = undistort(left)
und_left = rotate(und_left, "counterclock")
und_right = undistort(right)
und_right = rotate(und_right, "clock")
out = final(und_right, und_left)
cv2.imshow("und", cv2.resize(out, (800, 600)))
cv2.imwrite("und.jpg", cv2.resize(out, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
