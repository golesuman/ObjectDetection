import cv2 as cv
# lena=cv.imread('/home/suman/PycharmProjects/suman/lena.png')
cap=cv.VideoCapture(0)
cap.set(3,640)
cap.set(4,600)

classNames=[]
classfile='/home/suman/PycharmProjects/suman/coco.names'
with open(classfile,'rt') as f:
    classNames=f.read().rstrip('\n').split('\n')

configPath='/home/suman/PycharmProjects/suman/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath='/home/suman/PycharmProjects/suman/frozen_inference_graph.pb'
net = cv.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
    success,img=cap.read()
    classIds, confs, bbox=net.detect(img,confThreshold=0.5)
    print(classIds,bbox)
    if len(classIds)!=0:
        for classId ,confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv.rectangle(img,box,color=(0,255,0),thickness=2)
            cv.putText(img, classNames[classId-1].upper(),(box[0]+10,
                            box[1]+30),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),thickness=2)

        cv.imshow('Result',img
                  )
        cv.waitKey(1)