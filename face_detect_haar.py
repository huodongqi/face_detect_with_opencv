import cv2
import os

classifier = cv2.CascadeClassifier(
    '/Applications/anaconda3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
#读取内置分类器文件
color = (0, 255, 0)
#检测框颜色
def face_detect(image, filename):
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转换灰度，便于辨识
    faces = classifier.detectMultiScale(grey, scaleFactor=1.16,
                                        minNeighbors=5, minSize=(25, 25), flags=0)
    #人脸检测
    for face in faces:
        if len(face):
            x, y, w, h = face
            cv2.imwrite('./result_haar/' + filename,
                        cv2.rectangle(image, (x, y), (x + h, y + h), color, 2))
            #保存检测后的结果
    print(filename + ' have been saved!')

if 'result_haar' not in os.listdir('./'):
    os.mkdir('result_haar')
    #建立保存图像的文件夹
facedir = os.listdir('./data/')
#获取数据集列表
for faceimage in facedir:
    image = cv2.imread('./data/' + faceimage)
    face_detect(image, faceimage)
    # 数据集遍历并检测保存




