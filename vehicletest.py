import cv2
import numpy as np
import os
from tkinter import *
from tkinter import messagebox
import time
import matplotlib.pyplot as plt
from scipy.spatial import distance as distance
import cmath
from tkinter import filedialog
from gui_buttons import Buttons

# CREATING WINDOW USING TKINTER
root = Tk()
root.title(" OBJECT DETECTION MODEL")
# SETTING THE BACKGROUND AS LIGHT BLUE
root.configure(background="pink")
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

def camera():
    # initialize button
    button = Buttons()
    button.add_button("person", 20, 20)
    button.add_button("cell phone", 20, 100)
    button.add_button("remote", 20, 180)
    button.add_button("book", 20, 240)

    colors = button.colors

    # opencv dnn
    net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(320, 320), scale=1 / 255)

    # load class lists
    classes = []
    with open("dnn_model/classes.txt", "r") as file_object:
        for class_name in file_object.readlines():
            class_name = class_name.strip()
            classes.append(class_name)
    print("object list")
    print(classes)

    # initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # FULL HD 1920 X 1080

    # button_person = False
    def click_button(event, x, y, flags, params):
        global button_person
        if event == cv2.EVENT_LBUTTONDOWN:
            button.button_click(x, y)

    # create window
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", click_button)
    while True:
        ret, frame = cap.read()

        # get active button list
        active_buttons = button.active_buttons_list()
        print("active buttons", active_buttons)

        # object detection
        (class_ids, scores, bboxes) = model.detect(frame)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]

            if class_name in active_buttons:
                cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)

        # display button
        button.display_buttons(frame)

        # create button
        #  cv2.rectangle(frame, (20, 20), (220, 70), (0, 0, 200), -1)
        # polygon = np.array([[(20, 20), (220, 20), (220, 70), (20, 70)]])
        # cv2.fillPoly(frame, polygon, (0, 0, 200))
        #  cv2.putText(frame, "Person", (30, 60), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 3)

        # print("class ids", class_ids)
        # print("scores", scores)
        # print("bboxes", bboxes)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def videocheck():
    i = 0
    # DIALOG BOX TO SELECT VIDEO FILE
    fln = filedialog.askopenfilename(initialdir=os.getcwd(), title="Open file",
                                     filetypes=(("MP4", ".mp4"), ("All File", ".*")))
    videopath = fln

    # DEFINE A VIDEO CAPTURE OBJECT
    cap = cv2.VideoCapture('video.mp4')

    min_width_rect = 80
    min_height_rect = 80
    count_line_position = 550
    # initialize subtractor
    algo = cv2.createBackgroundSubtractorMOG2()

    def center_handle(x, y, w, h):
        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    detect = []
    offset = 6
    counter = 0

    while True:
        ret, frame1 = cap.read()
        if not ret:
            break
        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

        counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        for (i, c) in enumerate(counterShape):
            (x, y, w, h) = cv2.boundingRect(c)
            validate_counter = (w >= min_width_rect) and (h >= min_height_rect)
            if not validate_counter:
                continue

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "VEHICLE: " + str(counter), (x, y - 20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

            for (x, y) in detect:
                if y < (count_line_position + offset) and y > (count_line_position - offset):
                    counter += 1
                    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detect.remove((x, y))
                    print("Vehicle Counter:" + str(counter))

        cv2.putText(frame1, "VEHICLE COUNTER :" + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        # cv2.imshow('Detecter', dilatada)

        cv2.imshow('Video Original', frame1)

        if cv2.waitKey(1) == 13:
            break

    cv2.destroyAllWindows()
    cap.release()

# FRONT END
w2 = Label(root, justify=LEFT, text=" OBJECT DETECTION MODEL ")
w2.config(font=("elephant", 20), background="LIGHTblue")
w2.grid(row=1, column=0, columnspan=2, padx=100, pady=40)

NameLb = Label(root, text="PREDICT USING :")
NameLb.config(font=("elephant", 20), background="light blue")
NameLb.grid(row=13, column=0, pady=20)

lr = Button(root, text="Video", height=2, width=10, command=videocheck)
lr.config(font=("elephant", 15), background="light green")
lr.grid(row=15, column=0, pady=20)

lr = Button(root, text="Camera", height=2, width=10, command=camera)
lr.config(font=("elephant", 15), background="light green")
lr.grid(row=16, column=0, pady=20)

t3 = Text(root, height=2, width=15)
t3.config(font=(15))
t3.grid(row=15, column=1, padx=60)
t4 = Text(root, height=2, width=15)
t4.config(font=(15))
t4.grid(row=16, column=1, padx=60)

root.mainloop()
