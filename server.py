# flask 관련
from flask import Flask, g, make_response, jsonify, request, send_file
import time

# opencv , dlib, cmake 관련
import math
import numpy
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2

app = Flask(__name__)  # 앱 실행
app.debug = True
app.config['SERVER_NAME']
app.config['JSON_AS_ASCII'] = False  # 한글 오류 방지


@app.route("/")
def hello():
    return "This is test"


@app.route('/res', methods=['GET', 'POST'])
def getImg_and_sendResult():
    if request.method == 'POST':
        now = time.localtime()
        fname = str(now.tm_year) + "." + str(now.tm_mon) + "." + str(now.tm_mday) + " " + str(now.tm_hour) + "-" + str(
            now.tm_min) + "-" + str(now.tm_sec) + str(".jpg")
        f = request.files['file']
        f.save("./Graduate/static/images/" + fname)  # 서버에 이미지 저장

        # ----------------------------------------예측 부분 --------------------------------------------------------------------------------------
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        image = cv2.imread("./Graduate/static/images/" + fname)
        image = imutils.resize(image, width=500)

        def test(shape):

            # [x1,y1,x2,y2]
            testcaseA = [[19, 37, 24, 44],
                         [20, 38, 23, 43],
                         [21, 27, 22, 27],
                         [37, 41, 44, 46],
                         [38, 40, 43, 47],
                         [17, 27, 26, 27],
                         [19, 27, 24, 27],
                         [36, 50, 45, 52],
                         [0, 49, 16, 53],
                         [0, 48, 16, 54],
                         [3, 48, 13, 54],
                         [59, 6, 55, 10]]

            testcaseB = [[19, 41, 23, 47],
                         [20, 40, 24, 46],
                         [39, 31, 42, 35],
                         [0, 31, 16, 35],
                         [36, 50, 45, 52],
                         [0, 49, 16, 53],
                         [0, 48, 16, 54],
                         [36, 48, 45, 54],
                         [3, 48, 13, 54],
                         [59, 5, 55, 11],
                         [2, 48, 14, 54],
                         [61, 67, 63, 65]]

            testcaseC = [[31, 50, 35, 52],
                         [32, 50, 34, 52],
                         [2, 48, 14, 54],
                         [3, 48, 13, 54],
                         [4, 48, 12, 54],
                         [5, 48, 11, 54],
                         [6, 59, 10, 55],
                         [7, 58, 9, 56],
                         [48, 61, 54, 63],
                         [5, 59, 55, 11]]

            testcaseD = [[21, 27, 22, 27],
                         [19, 27, 24, 27],
                         [17, 27, 26, 27],
                         [20, 38, 23, 43],
                         [19, 37, 24, 44],
                         [17, 36, 26, 45],
                         [18, 36, 25, 45],
                         [38, 40, 43, 47],
                         [37, 41, 44, 46],
                         [40, 31, 47, 35],
                         [41, 48, 46, 54],
                         [36, 48, 45, 54]]

            # 뭔가 이상함
            testcaseE = [[21, 27, 22, 27],
                         [19, 27, 24, 27],
                         [17, 27, 26, 27],
                         [20, 38, 23, 43],
                         [19, 37, 24, 44],
                         [17, 27, 26, 27],
                         [18, 36, 26, 45],
                         [0, 31, 16, 35],
                         [0, 48, 16, 54],
                         [40, 31, 47, 35],
                         [41, 48, 46, 54],
                         [36, 48, 45, 54]]

            testcaseF = [[19, 37, 24, 44],
                         [20, 38, 23, 43],
                         [21, 27, 22, 27],
                         [37, 41, 44, 46],
                         [38, 40, 43, 47],
                         [17, 27, 26, 27],
                         [19, 27, 24, 27],
                         [17, 31, 26, 35],
                         [19, 31, 24, 35],
                         [21, 31, 22, 35]]

            left = list()
            right = list()
            distance = list()

            # 각 좌표의 거리를 구하고 선으로 그리기
            for (x1, y1, x2, y2) in testcaseF:
                left.append(euclidean_distance(shape[x1], shape[y1])), right.append(
                    euclidean_distance(shape[x2], shape[y2]))

            # 각 선에 대한 거리를 list에 저장하고 거리 출력
            for left_distance, right_distance in zip(left, right):
                d = abs(left_distance - right_distance)
                distance.append(d)
                print(str(left_distance) + " : " + str(right_distance) + " : " + str(d))

            # 거리의 평균 출력
            print("평균 : " + str(numpy.mean(distance)))

            # 거리가 3이 넘는다면 넘는 곳의 선을 그리기
            for d in distance:
                if d > 5:
                    cv2.line(image,
                             (shape[testcaseF[distance.index(d)][0]][0], shape[testcaseF[distance.index(d)][0]][1]),
                             (shape[testcaseF[distance.index(d)][1]][0], shape[testcaseF[distance.index(d)][1]][1]),
                             (0, 0, 255), 1)
                    cv2.line(image,
                             (shape[testcaseF[distance.index(d)][2]][0], shape[testcaseF[distance.index(d)][2]][1]),
                             (shape[testcaseF[distance.index(d)][3]][0], shape[testcaseF[distance.index(d)][3]][1]),
                             (0, 0, 255), 1)

        def show_raw_detection(image, detector, predictor, fname):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 그레이스케일 영상에서 얼굴 검출
            rects = detector(gray, 1)

            # 만약 얼굴이 검출되지 않았다면 rectangles[]이라면
            if not rects:
                print("얼굴이 검출되지 않았음")

            for (i, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                ##(x, y, w, h) = face_utils.rect_to_bb(rect)
                ##cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # show the face number
                ##cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image

                # 점의 논문의 좌표와 일치한지 확인
                num = 0
                for (x, y) in shape:
                    print("X:" + str(x) + " Y" + str(y))
                    cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
                    # 좌표의 번호를 입력하는 부분
                    # cv2.putText(image,str(num),(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

                    num = num + 1

                # 코등맨위부터 턱가운데까지 선긋기
                # cv2.line(image,shape[27],shape[33],(0,0,255),1)
                test(shape)

                # 이미지 저장
                cv2.imwrite("./Graduate/result/" + fname, image)

                # 이미지 출력
                cv2.imshow("Output", image)
                # cv2.waitKey(0)

        # 두점 사이의 거리를 구하는 유클리드 공식
        def euclidean_distance(shape1, shape2):
            x1 = shape1[0]
            y1 = shape1[1]
            x2 = shape2[0]
            y2 = shape2[1]

            result = round(math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)), 6)
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            print(result)
            return result

        # -----------------------------------------------------예측 끝 ------------------------------------------------------------------
        show_raw_detection(image, detector, predictor, fname)

        # 모델의 예측 결과를 폰으로 전송해줍니다.
        # return "전송성공!!!" # html이 아닌 텍스트 형태로 데이터 전송
        return send_file("result/" + fname, mimetype="image/jpg")

