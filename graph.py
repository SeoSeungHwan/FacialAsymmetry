import math

from imutils import face_utils
import pandas as pd
import imutils
import dlib
import cv2

all_distance=[]
#A일때
def test(shape,image):

    #[x1,y1,x2,y2]
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


    #뭔가 이상함
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

    #각 좌표의 거리를 구하고 선으로 그리기
    index=1;
    for(x1,y1,x2,y2) in testcaseF:
        left.append(euclidean_distance(shape[x1], shape[y1] ,index)), right.append(euclidean_distance(shape[x2], shape[y2],index))
        index = index+1


    #각 선에 대한 거리를 list에 저장하고 거리 출력
    i =1
    global result_json
    result_json = {}
    for left_distance,right_distance in zip(left , right):
        if left_distance>= right_distance:
            d = round(right_distance/left_distance,6)
        else:
            d = round(left_distance/right_distance,6)
        distance.append(d)
        i = i+1
    global all_distance
    all_distance.append(distance)
    #거리가 3이 넘는다면 넘는 곳의 선을 그리기
    for d in distance:
        if d <0.9:
            cv2.line(image, shape[testcaseF[distance.index(d)][0]], shape[testcaseF[distance.index(d)][1]], (0, 0, 255), 1)
            cv2.line(image, shape[testcaseF[distance.index(d)][2]], shape[testcaseF[distance.index(d)][3]], (0, 0, 255), 1)


def show_raw_detection(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #그레이스케일 영상에서 얼굴 검출
    rects = detector(gray, 1)

    #만약 얼굴이 검출되지 않았다면 rectangles[]이라면
    if not rects:
        print("얼굴이 검출되지 않았음")

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        #점의 논문의 좌표와 일치한지 확인
        num = 0
        for (x, y) in shape:
            cv2.circle(image, (x, y), 1, (0, 255,0 ), -1)

            num= num+1

        # 코등맨위부터 턱가운데까지 선긋기
        test(shape,image)



#두점 사이의 거리를 구하는 유클리드 공식
def euclidean_distance(shape1,shape2,index):
    x1 = shape1[0]
    y1 = shape1[1]
    x2 = shape2[0]
    y2 = shape2[1]

    result = round(math.sqrt(math.pow((x2-x1),2)+math.pow((y2-y1),2)),6)
    #각 선에 숫자쌍 표시
    if x1< x2:
        cv2.putText(image, str(index), ((int)(x1+(x2-x1)/2), (int)(y1+(y2-y1)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    else:
        cv2.putText(image, str(index), ((int)(x2 + (x1-x2) / 2), (int)(y1 + (y2 - y1) / 2)), cv2.FONT_HERSHEY_SIMPLEX,0.3, (255, 255, 255), 1)

    cv2.line(image, (x1,y1), (x2,y2), (255, 0, 0), 1)

    return result



# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# load the input image, resize it, and convert it to grayscale
for i in range(1,81):
    file = 'C:/Users/seunghwan/PycharmProjects/FacialAsymmetry/dataset/' + str(i) + '-6.jpg'
    print(file)
    image = cv2.imread(file)
    image = imutils.resize(image, width=500)
    show_raw_detection(image, detector, predictor)

print(all_distance)
df = pd.DataFrame.from_records(all_distance)
df.to_excel('test.xlsx')





