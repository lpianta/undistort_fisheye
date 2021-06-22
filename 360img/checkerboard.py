import cv2
import numpy as np

a = 6
b = 8

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + \
    cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)

objp = np.zeros((1, b*a, 3), np.float64)
objp[0, :, :2] = np.mgrid[0:a, 0:b].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.


img = cv2.imread("newleft3.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chess board corners
ret, corners = cv2.findChessboardCorners(gray, (a, b), None)


# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)

cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
imgpoints.append(corners)
objpoints = np.array(objpoints, dtype=np.float32)
imgpoints = np.array(imgpoints, dtype=np.float32)
# objpoints[:, :2] = np.mgrid[0:6, 0:8].T.reshape(-1, 2)
objpoints = objpoints.squeeze()
imgpoints = imgpoints.squeeze()
objpoints = np.array([objpoints], dtype=np.float32)
imgpoints = np.array([imgpoints], dtype=np.float32)


K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = np.zeros((1, 1, 3), dtype=np.float32)
tvecs = np.zeros((1, 1, 3), dtype=np.float32)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1], None, None)
print(gray.shape[::-1])
dim = img.shape[:2]

map1, map2 = cv2.initUndistortRectifyMap(
    mtx, dist, np.eye(3), mtx, dim, cv2.CV_16SC2)
undistorted_img = cv2.remap(
    img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

cv2.imwrite("undist_img.jpg", cv2.resize(undistorted_img, (800, 600)))
cv2.imshow("undist_img.jpg", cv2.resize(undistorted_img, (800, 600)))
cv2.waitKey(0)
cv2.destroyAllWindows()
