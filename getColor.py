import cv2
import numpy as np

# define range of blue color in HSV
color_dict_HSV = {'black': np.array([[180, 255, 30], [0, 0, 0]]),
                  'red': np.array([[180, 255, 255], [159, 50, 70]]),
                  'red2': np.array([[9, 255, 255], [0, 50, 70]]),
                  'green': np.array([[89, 255, 255], [36, 50, 70]]),
                  'blue': np.array([[128, 255, 255], [90, 50, 70]]),
                  'yellow': np.array([[35, 255, 255], [25, 50, 70]]),
                  'purple': np.array([[158, 255, 255], [129, 50, 70]]),
                  'orange': np.array([[24, 255, 255], [10, 50, 70]])}

# read image
orig = cv2.imread("images/colorTest.jpg")
#

# Get color by hue
hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
#

# For each color
for i in color_dict_HSV:
    mask = cv2.inRange(hsv, color_dict_HSV[i][1], color_dict_HSV[i][0])
    if i == "red":
        lower_mask = cv2.inRange(hsv, color_dict_HSV["red2"][1], color_dict_HSV["red2"][0])
        mask = lower_mask + mask
    elif i == "red2":
        continue

    colorMaskResult = cv2.bitwise_and(orig, orig, mask=mask)
    grayForColor = cv2.cvtColor(colorMaskResult, cv2.COLOR_BGR2GRAY)
    c, h = cv2.findContours(grayForColor, method=cv2.RETR_LIST, mode=cv2.CHAIN_APPROX_NONE)
    for x in c:
        area = cv2.contourArea(x)
        if area > 500:
            cv2.drawContours(orig, x, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(x, True)
            e = 0.01 * peri
            x1 = cv2.approxPolyDP(x, e, True)
            n = x1.ravel()
            x = n[0]
            y = n[1]
            # x, y, w, h = cv2.boundingRect(x1)
            text = i
            cv2.putText(orig, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
#

# # test for specific color
# color = "black"
# mask = cv2.inRange(hsv, color_dict_HSV[color][1], color_dict_HSV[color][0])
# colorMaskResult = cv2.bitwise_and(orig, orig, mask=mask)
# grayForColor = cv2.cvtColor(colorMaskResult, cv2.COLOR_BGR2GRAY)
# c, h = cv2.findContours(grayForColor, method=cv2.RETR_LIST, mode=cv2.CHAIN_APPROX_NONE)
# for x in c:
#     area = cv2.contourArea(x)
#     if area > 500:
#         cv2.drawContours(orig, x, -1, (255, 0, 0), 2)
#         peri = cv2.arcLength(x, True)
#         e = 0.01 * peri
#         x1 = cv2.approxPolyDP(x, e, True)
#         x, y, w, h = cv2.boundingRect(x1)
#         text = color
#         cv2.putText(orig, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
# #
cv2.imshow('Orig', orig)
cv2.waitKey(0)
