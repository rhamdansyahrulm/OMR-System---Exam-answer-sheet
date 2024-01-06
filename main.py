import cv2
import numpy as np
import utils

path = "image-test/1.png"
widthImg = 3580
heightImg = 2480

img = cv2.imread(path)

# PREPROCESSING
img = cv2.resize(img, (widthImg, heightImg))
imgContours = img.copy()
imgBiggestContours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)

# DEFINE CONTOURS
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 10)
rectCon = utils.rectContour(contours) # Finding a rectangle Contours
markingPoints = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[2])

if markingPoints.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (255, 0, 0), 50)

    markingPoints = utils.reorder(markingPoints)
    cv2.drawContours(imgBiggestContours, markingPoints, -1, (0, 255, 0), 50)
    pt1 = np.float32(markingPoints)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

imgBlank = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContours, imgBiggestContours, imgWarpColored, imgBlank])
imgStacked = utils.stackImages(imgArray, 0.5)

cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)