from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
from tkinter import messagebox as mb
root = tk.Tk()
t = Tk()



import cv2
import imutils
import numpy as np
import pyglet
from playsound import playsound
import matplotlib.pyplot as plt
from detect_people import detect_people_classes
from scipy.spatial import distance as dist
from detect import checkMask

def distance():
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
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread('more.jpg')
    # img = cv2.imread('less.jpg')
    video = cv2.VideoCapture("testvideo2.mp4")

    # loop over the frame
    while True:
        true, frame = video.read()

        # if frame where not grabbed break the loop
        if not true:
            break
        frame = imutils.resize(frame, width=700)
        results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

        violate = set()

        if len(results) >= 2:
            centroids = np.array([r[2] for r in results])
            w = ([r[3] for r in results])
            w_final = 0
            for o in w:
                w_final += o
            w_final = w_final / len(w)
            # print(w_final)
            D = dist.cdist(centroids, centroids, metric="euclidean")

            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):

                    if D[i, j] < 1.2 * w_final:
                        violate.add(i)
                        violate.add(j)
                        my_player.play()

        for (i, (prob, bbox, centroid, w, h)) in enumerate(results):
            (start_x, start_y, end_x, end_y) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            if i in violate:
                color = (0, 0, 255)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
            cv2.circle(frame, (cX, cY), 5, color, 1)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


def countPeople():
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
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    video = cv2.VideoCapture("test_video.mp4")

    # loop over the frame
    while True:
        true, frame = video.read()

        # if frame where not grabbed break the loop
        if not true:
            break
        frame = imutils.resize(frame, width=700)
        results = detect_people_classes(frame, net, ln, personIdx=labels.index("person"))

        for (i, (prob, bbox, centroid, w, h)) in enumerate(results):
            (start_x, start_y, end_x, end_y) = bbox
            (cX, cY) = centroid

            if len(results) >= 3:
                color = (0, 0, 255)
                my_player.play()
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), color, 2)
            # cv2.circle(frame, (cX, cY), 5, color, 1)

        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()

def detectMask():
    checkMask()



def quit():
    global root
    root.quit()

def callback():
    if mb.askyesno('Verify', 'Really quit?'):
        quit()
    else:
        mb.showinfo('No', 'Quit has been cancelled')


def home():
    f1 = Frame(root, width=600, height=700)
    f1.pack()

    u1 = Label(f1, text="COVID PRECAUTION PROJECT")
    u1.place(x=250, y=50)

    load = Image.open("background.jpeg")
    render = ImageTk.PhotoImage(load)
    img = Label(f1, image=render)
    img.image = render
    img.place(x=0, y=0)

    b1=Button(f1, text="Count of People", command=countPeople)
    b1.place(x=30, y=600, width= 180, height=40)
    b2 = Button(f1, text="Social Distancing", command=distance)
    b2.place(x=230, y=600, width=180, height=40)
    b3 = Button(f1, text="Mask Detection", command=detectMask)
    b3.place(x=430, y=600, width=180, height=40)
    b4 = Button(f1, text="QUIT", command=callback)
    b4.place(x=270, y=650, width=80, height=30)
    '''text = tk.Text(root, height=4, width=100)
    scroll = tk.Scrollbar(root)

    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text.pack(side=tk.LEFT, fill=tk.Y)

    scroll.config(command=text.yview)
    text.config(yscrollcommand=scroll.set)

    trend = "PREVENTION IS BETTER THAN CURE! STAY HOME STAY SAFE!  "
    text.insert(tk.END, trend)'''

home()
t.mainloop()
















