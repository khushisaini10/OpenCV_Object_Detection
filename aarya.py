import numpy as np
import cv2
import os
from tkinter import *
from tkinter import messagebox
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as distance
import cmath
from tkinter import filedialog

# CREATING WINDOW USING TKINTER
root = Tk()
root.title(" OBJECT DETECTION MODEL")
# SETTING THE BACKGROUND AS LIGHT BLUE
root.configure(background="lightblue")
# ACCESSING CLASSE STORED IN COCONAMES
labelpath = 'coco.names'
file = open(labelpath)
# LOADING THE NAME OF CLASSES
label = file.read().strip().split("\n")
label[0]

# LOADING PRE-TRAINED MODEL WEIGHTS AND CONFIGURATIONS
weightspath = 'yolov4.weights'
configpath = 'yolov4.cfg'

# LOADING MODEL AND ITS WEIGHTS STORED IN NET OBJECT
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")

# USING NET OBJ,GETLAYERSNAME RETURN THE LIST OF STRING
layer_names = net.getLayerNames()
# net.getUnconnectedOutLayers() gives the position of the layers.
# the output is an ndarray(multidimensional container of items of the same type and size)
# of shape (1,). So to get the integer we do ln[0].
# And to get the index we subtract 1 from the position.
ln = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def videocheck():
    i = 0
    # DIALOG BOX TO SELECT VIDEO FILE
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open file",
                                     filetypes=(("MP4", ".mp4"), ("All File", ".*")))
    videopath = fln

    # DEFINE A VIDEO CAPTURE OBJECT
    video = cv2.VideoCapture(videopath)
    ret = video

    # INFINITE LOOP TO READ THE FRAMES USING VIDEO OBJECT
    data = []
    while (True):
        ret, frame = video.read()
        if ret == False:
            print('Error running the file :(')
        frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)

        # BLOBFROMIMAGE RETURN A 4D ARRAY FOR THE INPUT IMAGE
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        r = blob[0, 0, :, :]

        # IDENTIFY THE OBJECT AND SET NEW VALUE TO THE NETWORK
        net.setInput(blob)

        # STORE DETECTED OBJECTS WHICH CONSIST OF CLASS LABEL, CONFIDENCE STORE, COORDINATES
        outputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []
        center = []
        output = []
        count = 0
        results = []
        # STORE HEIGHT AND WIDTH OF THE IMAGE
        h, w = frame.shape[:2]
        for output in outputs:
            for detection in output:
                # EXTRACT SCORES-ELEMENTS FROM 5TH INDEX TO LAST
                scores = detection[5:]
                # EXTRACT CLASS ID-MAX ELEMENT INDEX OF SCORES
                classID = np.argmax(scores)
                # EXTRACT CONFIDENCE SCORE FOR CLASS
                confidence = scores[classID]

                if confidence > 0.5:
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    # X,Y ARE THE TOP LEFT COORDINATES
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # UPDATING CENTROID OF THE BOUNDING BOX
                    center.append((centerX, centerY))
                    # ASSIGNING NEW COORDINATES OF THE BOX
                    box = [x, y, int(width), int(height)]
                    # APPEND COORDINATES OF THE BOUNDING BOX
                    boxes.append(box)
                    # APPEND CONFIDENCE OF THE SELECTED BOUNDED BOX
                    confidences.append(float(confidence))
                    # APPEND CLASSID OF THE SELECTED BOUNDED BOX
                    classIDs.append(classID)

        # MOST PROMISING BOUNDED BOXES ARE STORED IN INDICES CONFIDENCE<0.4 AND INTERACTION OF UNION>0.5
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                # EXTRACT COORDINATES(TOP LEFT) X,Y AND WIDTH AND HEIGHT OF BOUNDED BOXES
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # OUT OF SO MANY CLASS IDS SELECTING ONLY THOSE WHERE CLASS ID = PERSON
                if (label[classIDs[i]] == 'person'):
                    # people()
                    # OBJECTS DETECTED OF PARTICULA CLASS LABEL ARE STORED HERE
                    cX = (int)(x + (y / 2))
                    cY = (int)(w + (h / 2))
                    # UPDATING CENTROID COORDINTES
                    center.append((cX, cY))
                    res = ((x, y, x + w, y + h), center[i])
                    results.append(res)
                    # CALCULATING EUCLEDIAN DISTANCE
                    dist = cmath.sqrt(
                        ((center[i][0] - center[i + 1][0]) * 2) + ((center[i][1] - center[i + 1][1]) * 2))
                    # IF DISTANCE BETWEEN CENTROIDS IS <100 THEN WE CONSIDE IT AS UNSAFE
                    if (dist.real < 100):
                        # CENTROID ARE CLOSE SO COLOR IS RED BGR(0,0,255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        # CENTROID DISPLAYED USING CIRCLE
                        cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                        # DISTANCE BETWEEN CENTROID DISPLAYED USING LINE
                        cv2.line(frame, (center[i][0], center[i][1]), (center[i + 1][0], center[i + 1][1]), (0, 0, 255),
                                 thickness=3, lineType=8)
                        # INCREMENTING THE COUNT OF PEOPLE
                        count = count + 1

                    else:
                        # CENTROID ARE FAR SO COLOR IS GREEN BGR(0,255,0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # CENTROID DISPLAYED USING CIRCLE
                        cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                        # INCREMENTING THE COUNT OF PEOPLE
                        count = count + 1

            # DISPLAYING TOTAL COUNT
            cv2.putText(frame, "Count: {}".format(
                count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        # DISPLAYING FRAME WITH BOUNDED BOXES
        cv2.imshow('Frame', frame)
        print(count)
        current_time = i
        # STORING COUNT OF PEOPLE AND TIME IN ORDER TO PLOT THE GRAPH
        data.append((count, current_time))
        i = i + 1
        # PRESS Q TO EXIT
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print(1)
    # AFTER LOOP FINISHES RELEASE THE OBJECT
    video.release()
    # DESTRYING THE OPEN WINDOW
    cv2.destroyAllWindows()
    # IF COUNT > 0 DISPLAY IT IN WINDOW
    if count > 0:
        t3.delete("1.0", END)
        t3.insert(END, count)

    # STORING THE SORTED DATA(COUNT,TIME)
    def Sort(sub_li):
        sub_li.sort(key=lambda x: x[1])
        return sub_li

    # Driver Code
    # PRINTING THE SORTED DATA(TIME,COUNT)
    print(Sort(data))
    print(data)
    # STORING COUNT IN Y AND TIME IN X
    x = []
    y = []
    for i in data:
        x.append(i[1])
    for i in data:
        y.append(i[0])

    # plotting the points
    plt.plot(x, y)

    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')

    # giving a title to my graph
    plt.title('DENSITY GRAPH!')

    # function to show the plot
    plt.show()


def photo():
    ret = True
    f_types = [('Image Files', '*.jpg')]
    # DIALOG BOX TO SELECT PHOTO
    filename = filedialog.askopenfilename(filetypes=f_types)
    # READING THE IMAGE
    img = cv2.imread(filename)
    frame = img
    # DISPLAYING IMAGE WITH FRAMES
    cv2.imshow('Frame', frame)
    if ret == False:
        print('Error running the file :(')
    # RESIZING THE FRAME
    frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
    # BLOBFROMIMAGE RETURN A 4D ARRAY FOR THE INPUT IMAGE
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    r = blob[0, 0, :, :]
    # IDENTIFY THE OBJECT AND SET NEW VALUE TO THE NETWORK
    net.setInput(blob)
    # STORE DETECTED OBJECTS WHICH CONSIST OF CLASS LABEL, CONFIDENCE STORE, COORDINATES
    outputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []
    center = []
    output = []
    count = 0
    results = []

    # STORE HEIGHT AND WIDTH OF THE IMAGE
    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            # EXTRACT SCORES-ELEMENTS FROM 5TH INDEX TO LAST
            scores = detection[5:]
            # EXTRACT CLASS ID-MAX ELEMENT INDEX OF SCORES
            classID = np.argmax(scores)
            # EXTRACT CONFIDENCE SCORE FOR CLASS
            confidence = scores[classID]

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # X,Y ARE THE TOP LEFT COORDINATES
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # UPDATING CENTROID OF THE BOUNDING BOX
                center.append((centerX, centerY))
                # ASSIGNING NEW COORDINATES OF THE BOX
                box = [x, y, int(width), int(height)]
                # APPEND COORDINATES OF THE BOUNDING BOX
                boxes.append(box)
                # APPEND CONFIDENCE OF THE SELECTED BOUNDED BOX
                confidences.append(float(confidence))
                # APPEND CLASSID OF THE SELECTED BOUNDED BOX
                classIDs.append(classID)

    # MOST PROMISING BOUNDED BOXES ARE STORED IN INDICES CONFIDENCE<0.4 AND INTERACTION OF UNION>0.5
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            # EXTRACT COORDINATES(TOP LEFT) X,Y AND WIDTH AND HEIGHT OF BOUNDED BOXES
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # OUT OF SO MANY CLASS IDS SELECTING ONLY THOSE WHERE CLASS ID = PERSON
            if (label[classIDs[i]] == 'person'):
                # people()
                # OBJECTS DETECTED OF PARTICULA CLASS LABEL ARE STORED HERE
                cX = (int)(x + (y / 2))
                cY = (int)(w + (h / 2))

                # UPDATING CENTROID COORDINTES
                center.append((cX, cY))
                res = ((x, y, x + w, y + h), center[i])
                results.append(res)
                # CALCULATING EUCLEDIAN DISTANCE
                dist = cmath.sqrt(
                    ((center[i][0] - center[i + 1][0]) * 2) + ((center[i][1] - center[i + 1][1]) * 2))
                # IF DISTANCE BETWEEN CENTROIDS IS <100 THEN WE CONSIDE IT AS UNSAFE
                if (dist.real < 100):
                    # CENTROID ARE CLOSE SO COLOR IS RED BGR(0,0,255)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # CENTROID DISPLAYED USING CIRCLE
                    cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                    # DISTANCE BETWEEN CENTROID DISPLAYED USING LINE
                    cv2.line(frame, (center[i][0], center[i][1]), (center[i + 1][0], center[i + 1][1]), (0, 0, 255),
                             thickness=3, lineType=8)
                    # INCREMENTING THE COUNT OF PEOPLE
                    count = count + 1

            else:
                # CENTROID ARE FAR SO COLOR IS GREEN BGR(0,255,0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # CENTROID DISPLAYED USING CIRCLE
                cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                # INCREMENTING THE COUNT OF PEOPLE
                count = count + 1

        # DISPLAYING TOTAL COUNT
        cv2.putText(frame, "Count: {}".format(
            count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    # DISPLAYING FRAME WITH BOUNDED BOXES
    cv2.imshow('Frame', frame)
    # IF COUNT > 0 DISPLAY IT IN WINDOW
    if count > 0:
        t4.delete("1.0", END)
        t4.insert(END, count)
    cv2.waitKey()
    # DESTROY ALL WINDOW
    cv2.destroyAllWindows()


# STORING THE SORTED DATA(COUNT,TIME)
def Sort(sub_li):
    sub_li.sort(key=lambda x: x[1])
    return sub_li


# FRONT END
w2 = Label(root, justify=LEFT, text=" OBJECT DETECTION MODEL ")
w2.config(font=("elephant", 20), background="LIGHTblue")
w2.grid(row=1, column=0, columnspan=2, padx=100, pady=40)

NameLb = Label(root, text="PREDICT USING :")
NameLb.config(font=("elephant", 20), background="Lightblue")
NameLb.grid(row=13, column=0, pady=20)

lr = Button(root, text="Video", height=2, width=10, command=videocheck)
lr.config(font=("elephant", 15), background="light green")
lr.grid(row=15, column=0, pady=20)

lr = Button(root, text="Camera", height=2, width=10, command=photo)
lr.config(font=("elephant", 15), background="lightgreen")
lr.grid(row=16, column=0, pady=20)

t3 = Text(root, height=2, width=15)
t3.config(font=(15))
t3.grid(row=15, column=1, padx=60)
t4 = Text(root, height=2, width=15)
t4.config(font=(15))
t4.grid(row=16, column=1, padx=60)

root.mainloop()