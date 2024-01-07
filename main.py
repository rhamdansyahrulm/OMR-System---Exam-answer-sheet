import cv2
import numpy as np
import utils

path = "image-test/2.png"
widthImg = 3580
heightImg = 2480
question = 50
choices = 4
ans = [0, 1, 0, 0, 2, 3, 2, 3, 1, 1, 0, 3, 0, 0, 1, 3, 2, 1, 1, 2, 2, 0, 0, 3, 1, 2, 2, 3, 0, 2, 0, 3, 0, 1, 3, 2, 0, 3, 0, 1, 1, 0, 3, 0, 2, 1, 0, 2, 1, 0]

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

    gradePoints = utils.reorder(gradePoints)
    cv2.drawContours(imgBiggestContours, gradePoints, -1, (0, 255, 0), 50)
    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
    imgGradedColored = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grades", imgGradedColored)

    #APPLY THRESHOLD
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1]

    split_boxes = utils.splitBoxes(imgThresh)
    # cv2.imshow("Option A", split_boxes[2])
    # print(cv2.countNonZero(split_boxes[0]), cv2.countNonZero(split_boxes[1]), cv2.countNonZero(split_boxes[2]), cv2.countNonZero(split_boxes[3]))

    myPixelVal = np.zeros((question, choices))
    countR = 0
    countC = 0

    for image in split_boxes:
        totalixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0

    mark_choices = []
    for question_no in range(0, question):
        arr = myPixelVal[question_no]
        mark_choice = np.where(arr==np.amax(arr))
        mark_choices.append(mark_choice[0][0])

    print(mark_choices)
    #Grading
    grading = []
    for no in range(0, question):
        if ans[no] == mark_choices[no]:
            grading.append(1)
        else : grading.append(0)

    score = (sum(grading)/ question) * 100
    print(score)

    #DISPLAYING ANSWER
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, mark_choices, grading, ans)
    imRawDrawing = np.zeros_like(imgWarpColored)
    imRawDrawing = utils.showAnswers(imRawDrawing, mark_choices, grading, ans)
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (widthImg, heightImg))
    # cv2.imshow("test", imgResult)

imgBlank = np.zeros_like(img)
imgArray = ([img, imgGray, imgBlur, imgCanny],
            [imgContours, imgBiggestContours, imgWarpColored, imgResult])
imgStacked = utils.stackImages(imgArray, 0.5)

cv2.imshow("Stacked Images", imgStacked)
cv2.waitKey(0)