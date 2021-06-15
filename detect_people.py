import cv2
import numpy as np
min_cof = 0.3
min_thr = 0.3

def detect_people_classes(frame, net, ln, personIdx=0):

    (h, w) = frame.shape[:2]

    results = []

   
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(ln)

    boxes = []
    centroids = []
    confidences = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            
            if classID == personIdx and confidence > min_cof:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")

                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                centroids.append((center_x, center_y))
                confidences.append(float(confidence))

    
    idx = cv2.dnn.NMSBoxes(boxes, confidences, min_cof, min_thr)

   
    if len(idx) > 0:
        for i in idx.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            r = (confidences[i], (x, y, x+w, y+h), centroids[i],w,h)
            results.append(r)

    return results














