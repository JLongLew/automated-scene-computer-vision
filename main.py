import cv2
import numpy as np

# kernel for morphing
k = np.ones((3, 3), np.uint8)

# Area of objects
areaOfObjects = []

# read image
orig = cv2.imread("images/shapes.png")
color = cv2.imread("images/shapes.png")
#

# blur and grayscale
blur = cv2.GaussianBlur(orig, (3, 3), 2)
gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
#

# define range of blue color in HSV
color_dict_HSV = {'black': np.array([[180, 255, 30], [0, 0, 0]]),
                  'red': np.array([[180, 255, 255], [159, 50, 70]]),
                  'red2': np.array([[9, 255, 255], [0, 50, 70]]),
                  'green': np.array([[89, 255, 255], [36, 50, 70]]),
                  'blue': np.array([[128, 255, 255], [90, 50, 70]]),
                  'yellow': np.array([[35, 255, 255], [25, 50, 70]]),
                  'purple': np.array([[158, 255, 255], [129, 50, 70]]),
                  'orange': np.array([[24, 255, 255], [10, 50, 70]])}

# Get color by hue
hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)

# Get user input for background type
while 1:
    threshType = input("1:Black background, 2:White background, 3:Non Specific background: ")
    if threshType == "1":
        threshType = cv2.THRESH_BINARY
        threshVal = 75
        break
    elif threshType == "2":
        threshType = cv2.THRESH_BINARY_INV
        threshVal = 240
        break
    elif threshType == "3":
        threshType = cv2.THRESH_BINARY_INV
        threshVal = 100
        break
    else:
        continue
#

# thresholding, morphology, and find contours
r, thresh = cv2.threshold(gray, threshVal, 255, threshType)
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=k, iterations=1)
c, h = cv2.findContours(morph, method=cv2.RETR_LIST, mode=cv2.CHAIN_APPROX_NONE)
#

# loop through each contour
i = 0  # counter to keep track of each contour
for x in c:
    area = cv2.contourArea(x)
    if area > 500:
        cv2.drawContours(orig, x, -1, (255, 0, 0), 2)
        i = i + 1
        areaOfObjects.append(area)
        peri = cv2.arcLength(x, True)
        e = 0.01 * peri
        x1 = cv2.approxPolyDP(x, e, True)
        print("Number of corners for object", i)
        objCorners = len(x1)

        # get the coordinates of the and approximate rectangle of the object
        # with the coordinates we can put text label on to the objects
        x, y, w, h = cv2.boundingRect(x1)

        print(objCorners)
        print("Shape:")
        if objCorners == 3:
            objectType = "Triangle"
        elif objCorners == 4:
            aspRatio = w / float(h)  # check the ratio between width and height
            if 0.98 < aspRatio < 1.03:
                objectType = "Square"
            else:
                objectType = "Rectangle"
        elif objCorners > 10:
            objectType = "Circle"
        else:
            objectType = "None"

        # add the shape label to the objects
        text = str(i) + ":" + objectType
        cv2.putText(orig, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))

        print(objectType)
        print("")

# For each color
for i in color_dict_HSV:
    mask = cv2.inRange(hsv, color_dict_HSV[i][1], color_dict_HSV[i][0])
    if i == "red":
        lower_mask = cv2.inRange(hsv, color_dict_HSV["red2"][1], color_dict_HSV["red2"][0])
        mask = lower_mask + mask
    elif i == "red2":
        continue

    colorMaskResult = cv2.bitwise_and(color, color, mask=mask)
    grayForColor = cv2.cvtColor(colorMaskResult, cv2.COLOR_BGR2GRAY)
    c, h = cv2.findContours(grayForColor, method=cv2.RETR_LIST, mode=cv2.CHAIN_APPROX_NONE)
    for x in c:
        area = cv2.contourArea(x)
        if area > 500:
            cv2.drawContours(color, x, -1, (255, 0, 0), 2)
            peri = cv2.arcLength(x, True)
            e = 0.01 * peri
            x1 = cv2.approxPolyDP(x, e, True)
            n = x1.ravel()
            x = n[0]
            y = n[1]
            # x, y, w, h = cv2.boundingRect(x1)
            text = i
            cv2.putText(color, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
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
cv2.waitKey(0)

# Get the smallest and largest object
max_value = max(areaOfObjects)
max_index = areaOfObjects.index(max_value)
min_value = min(areaOfObjects)
min_index = areaOfObjects.index(min_value)
#

# Print the number of objects
print("The number of objects in the scene is:")
print(len(areaOfObjects))
print("")
#

# Print the id for largest and smallest objects in the scene
print("Largest object in the scene is id: ", max_index + 1, "Smallest object in the scene is id: ", min_index + 1)
#

# Show the images and its form after operations
cv2.imshow("Orig", orig)
cv2.imshow("Color", color)
cv2.imshow("Blur", blur)
cv2.imshow("Thresh", thresh)
#

cv2.waitKey(0)
