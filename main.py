import os
import cvzone
from cvzone.ClassificationModule import Classifier
import cv2


cap = cv2.VideoCapture(1)
Classifier = Classifier('Resources/Model/keras_model.h5', 'Resources/Model/labels.txt')

imgArrow = cv2.imread('Resources/arrow.png', cv2.IMREAD_UNCHANGED)

binClassID = 0

imgObjList = []
pathFolder = "Resources/Objects"
pathList = os.listdir(pathFolder)
for path in pathList:
    imgObjList.append(cv2.imread(os.path.join(pathFolder, path), cv2.IMREAD_UNCHANGED))

imgBinList = []
pathFolder = "Resources/Bins"
pathList = os.listdir(pathFolder)
for path in pathList:
    imgBinList.append(cv2.imread(os.path.join(pathFolder, path), cv2.IMREAD_UNCHANGED))

classDic = {0: None,
            1: 0,
            2: 3,
            3: 0,
            4: 0,
            5: 3,
            6: 2,
            7: 1}

while True:
    _, img = cap.read()

    imgResize = cv2.resize(img, (454, 340))
    imgBackground = cv2.imread('Resources/background.png')

    prediction = Classifier.getPrediction(img)
    imgClassID = prediction[1]

    if imgClassID != 0:
        imgBackground = cvzone.overlayPNG(imgBackground, imgObjList[imgClassID-1], (909, 127))
        imgBackground = cvzone.overlayPNG(imgBackground, imgArrow, (978, 320))
        binClassID = classDic[imgClassID]
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinList[binClassID], (895, 374))
    else:
        imgBackground = cvzone.overlayPNG(imgBackground, imgBinList[3], (895, 374))

    imgBackground[148:148+340, 159:159+454] = imgResize

    cv2.imshow("Output", imgBackground)
    cv2.waitKey(1)
