import math

import numpy
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

#A일때
def test(shape):

    testcaseA = [[19,37,24,44],
                [20,38,23,43],
                [21,27,22,27],
                [37,41,44,46],
                [38,40,43,47],
                [17,27,26,27],
                [19,27,24,27],
                [36,50,45,52],
                [0,49,16,53],
                [0,48,16,54],
                [3,48,13,54],
                [59,6,55,10]]

    left = list()
    right = list()
    distance = list()

    #각 좌표의 거리를 구하고 선으로 그리기
    for(x1,y1,x2,y2) in testcaseA:
        left.append(euclidean_distance(shape[x1], shape[y1])), right.append(euclidean_distance(shape[x2], shape[y2]))

    #각 선에 대한 거리를 list에 저장하고 거리 출력
    for left_distance,right_distance in zip(left , right):
        d = abs(left_distance-right_distance)
        distance.append(d)
        print(str(left_distance)+" : " +str(right_distance) + " : " + str(d))

    #거리의 평균 출력
    print("평균 : " + str(numpy.mean(distance)))

    #거리가 3이 넘는다면 넘는 곳의 선을 그리기
    for d in distance:
        if d > 3:
            cv2.line(image, shape[testcaseA[distance.index(d)][0]], shape[testcaseA[distance.index(d)][1]], (0, 0, 255), 1)
            cv2.line(image, shape[testcaseA[distance.index(d)][2]], shape[testcaseA[distance.index(d)][3]], (0, 0, 255), 1)


def show_raw_detection(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #그레이스케일 영상에서 얼굴 검출
    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        #determine the facial landmarks for the face region, then
        #convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # convert dlib's rectangle to a OpenCV-style bounding box
        #[i.e., (x, y, w, h)], then draw the face bounding box
        ##(x, y, w, h) = face_utils.rect_to_bb(rect)
        ##cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # show the face number
        ##cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        
        #점의 논문의 좌표와 일치한지 확인
        num = 0
        for (x, y) in shape:
            print("X:" +str(x) +" Y" + str(y))
            cv2.circle(image, (x, y), 1, (0, 255,0 ), -1)
            #좌표의 번호를 입력하는 부분
            #cv2.putText(image,str(num),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            num= num+1

        # 코등맨위부터 턱가운데까지 선긋기
        #cv2.line(image,shape[27],shape[33],(0,0,255),1)
        test(shape)
        cv2.imshow("Output", image)
        cv2.waitKey(0)

#두점 사이의 거리를 구하는 유클리드 공식
def euclidean_distance(shape1,shape2):
    x1 = shape1[0]
    y1 = shape1[1]
    x2 = shape2[0]
    y2 = shape2[1]

    result = round(math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2)),6)
    cv2.line(image, shape1, shape2, (255, 0, 0), 1)
    print(result)
    return result

def draw_individual_detections(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        #convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)
            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

            # show the particular face part
            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)
        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(image, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
image = cv2.imread('C:/Users/tmdgh/PycharmProjects/FacialAsymmetry/realdata2.jpg')
image = imutils.resize(image, width=500)
show_raw_detection(image, detector, predictor)
#draw_individual_detections(image, detector, predictor)






