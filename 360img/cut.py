import cv2
import os

img_list = [f for f in os.listdir(
    "/media/lpianta/Dati/code/fisheye_calibration/360img") if os.path.isfile(os.path.join("/media/lpianta/Dati/code/fisheye_calibration/360img", f))]
img_list = [i for i in img_list if i.__contains__(".jpg")]
counter = 0
for i in img_list:
    print(i)
    img = cv2.imread("360img/" + i)
    height, width, _ = img.shape
    left_img = img[0:height, 0:width//2]
    right_img = img[0:height, width//2:width]
    cv2.imwrite(f"newleft{counter}.jpg", left_img)
    cv2.imwrite(f"newright{counter}.jpg", right_img)
    counter += 1
