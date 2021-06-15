import cv2
import winsound

font = cv2.FONT_HERSHEY_COMPLEX

video = cv2.VideoCapture('pedestrians1.mp4')


pedestrian_tracker_file = 'haarcascade_fullbody.xml'


pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


while True:

    (read_successful, frame) = video.read()
    if read_successful:
        resized = cv2.resize(frame, (1000, 700))
    else:
        break


    pedestrians = pedestrian_tracker.detectMultiScale(resized)
    count = 0
    x=10
    y=50
    print(pedestrians)
    for val in pedestrians:
        count += 1
    cv2.putText(resized, "Number of people detected "+str(count), (x, y), font, 1, (244, 250, 250), 2)
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #cv2.rectangle(resized, (x, y), (x + w, y+0), (255, 0, 0), 2)

    '''count = 0
    for val in pedestrians:
        count += 1'''
    #print("oh          yeah          this         is         the      count     maan!!!!!!",count)
    if count >= 5:
        for (x, y, w, h) in pedestrians:

            cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 0, 255), 2)

        frequency = 2500  # Set Frequency To 2500 Hertz
        duration = 1000  # Set Duration To 1000 ms == 1 second
        winsound.Beep(frequency, duration)

    cv2.imshow('Person detected', resized)
    cv2.waitKey(1)



print("code completed")