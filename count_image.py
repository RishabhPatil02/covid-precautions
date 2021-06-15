import cv2
import imutils
import numpy as np

from detect_people import detect_people_classes
from scipy.spatial import distance as dist


labels = open("coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

#img = cv2.imread('more_count.jpg')
img = cv2.imread('less_count.jpg')



while True:
    frame = imutils.resize(img, width= 700)
    results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

    

    

    for (i, (prob, bbox, centroid,w,h)) in enumerate(results):
        (start_x, start_y, end_x, end_y) = bbox
        (cX, cY) = centroid

        if  len(results) >=6:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        # cv2.circle(frame, (cX, cY), 5, color, 1)


    cv2.imshow("frame", frame)
    key =cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
















