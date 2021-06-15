import cv2
import imutils
import numpy as np
from playsound import playsound
from detect_people import detect_people_classes
from scipy.spatial import distance as dist

import pyglet				
pyglet.options['search_local_libs'] = True

my_music = pyglet.media.load("count.mp3")	

my_player = pyglet.media.Player()

my_player.queue(my_music)
my_player.loop = False
	

labels = open("coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

video = cv2.VideoCapture("test_video.mp4")

# loop over the frame
while True:
    true, frame= video.read()

    # if frame where not grabbed break the loop
    if not true:
        break
    frame = imutils.resize(frame, width= 700)
    results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

    

    

    for (i, (prob, bbox, centroid,w,h)) in enumerate(results):
        (start_x, start_y, end_x, end_y) = bbox
        (cX, cY) = centroid

        if  len(results) >=3:
            color = (0, 0, 255)
            my_player.play()
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        # cv2.circle(frame, (cX, cY), 5, color, 1)


    cv2.imshow("frame", frame)
    key =cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
















