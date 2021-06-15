import cv2
import imutils
import numpy as np

from detect_people import detect_people_classes
from scipy.spatial import distance as dist
import pyglet				
	

pyglet.options['search_local_libs'] = True

my_music = pyglet.media.load("distance.mp3")
my_player = pyglet.media.Player()

my_player.queue(my_music)
my_player.loop = False

labels = open("coco.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0]-1] for i in net.getUnconnectedOutLayers()]

img = cv2.imread('more.jpg')
#img = cv2.imread('less.jpg')
video = cv2.VideoCapture("testvideo2.mp4")

# loop over the frame
while True:
    true, frame= video.read()


    # if frame where not grabbed break the loop
    if not true:
        break
    frame = imutils.resize(frame, width= 700)
    results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

    violate = set()

    if len(results) >= 2:
        centroids = np.array([r[2] for r in results])
        w=([r[3] for r in results])
        w_final=0
        for o in w:
            w_final+=o
        w_final=w_final/len(w)
        # print(w_final)
        D = dist.cdist(centroids, centroids, metric="euclidean")

        for i in range(0, D.shape[0]):
            for j in  range(i+1, D.shape[1]):

                if D[i, j]  < 1.2*w_final:
                    
                    violate.add(i)
                    violate.add(j)
                    my_player.play()

    for (i, (prob, bbox, centroid,w,h)) in enumerate(results):
        (start_x, start_y, end_x, end_y) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        if  i in violate:
            color = (0, 0, 255)
            

        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)


    cv2.imshow("frame", frame)
    key =cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
















