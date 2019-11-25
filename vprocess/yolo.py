#!/usr/bin/env python3.5
import os
import dlib
import numpy as np
import cv2
import time
import darknet
import sys
from ctypes import *
import math
from datetime import datetime
import random

class Person:
    def __init__(self, name, t_from, t_to, x, y, w, h, prob):
        self.name = name
        self.t_from = t_from
        self.t_to = t_to
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.prob = prob

    def show(self):
        print ("name:", self.name)
        print ("prob:", self.prob)
        print ("from:", self.t_from)
        print ("to:", self.t_to)
        print ("x:", self.x)
        print ("y:", self.y)
        print ("w:", self.w)
        print ("h:", self.h)

    def union(self, b):
        x = min(self.x, b.x)
        y = min(self.y, b.y)
        w = max(self.x+self.w, b.x+b.w) - x
        h = max(self.y+self.h, b.y+b.h) - y
        return (x, y, w, h)

    def intersection(self, b):
        x = max(self.x, b.x)
        y = max(self.y, b.y)
        w = min(self.x+self.w, b.x+b.w) - x
        h = min(self.y+self.h, b.y+b.h) - y
        if w<0 or h<0: return (0, 0, 0, 0) # or (0,0,0,0) ?
        return (x, y, w, h)

    def area(self):
        return (self.w * self.h)

    def area_intersection(self, b):
        (x, y, w, h) = self.intersection(b)
        return (w * h)



class YOLO_NN:
    def __init__(self, path='.', display=True):

        self.display = display

        self.configPath = path + "/cfg/yolov3-tiny.cfg"
        self.weightPath = path + "/yolov3-tiny.weights"
        #self.configPath = path + "/yolov3/yolov3.cfg"
        #self.weightPath = path + "/yolov3/yolov3.weights"
        self.metaPath = path + "/cfg/coco.data"

        print("self.configPath: " + self.configPath)
        print("self.weightPath: " + self.weightPath)
        print("self.metaPath: " +  self.metaPath)

        self.netMain = None
        self.metaMain = None
        self.altNames = None

        if not os.path.exists(self.configPath):
            raise ValueError("Invalid config path `" +
                             os.path.abspath(self.configPath)+"`")
        if not os.path.exists(self.weightPath):
            raise ValueError("Invalid weight path `" +
                             os.path.abspath(self.weightPath)+"`")
        if not os.path.exists(self.metaPath):
            raise ValueError("Invalid data file path `" +
                             os.path.abspath(self.metaPath)+"`")
        if self.netMain is None:
            self.netMain = darknet.load_net_custom(self.configPath.encode(
                "ascii"), self.weightPath.encode("ascii"), 0, 1)  # batch size = 1
        if self.metaMain is None:
            self.metaMain = darknet.load_meta(self.metaPath.encode("ascii"))
        if self.altNames is None:
            try:
                with open(self.metaPath) as metaFH:
                    metaContents = metaFH.read()
                    import re
                    match = re.search("names *= *(.*)$", metaContents,
                                      re.IGNORECASE | re.MULTILINE)
                    if match:
                        result = match.group(1)
                    else:
                        result = None
                    try:
                        if os.path.exists(result):
                            with open(result) as namesFH:
                                namesList = namesFH.read().strip().split("\n")
                                self.altNames = [x.strip() for x in namesList]
                    except TypeError:
                        pass
            except Exception:
                pass

        # Create an image we reuse for each detect
        self.darknet_width = darknet.network_width(self.netMain)
        self.darknet_height = darknet.network_height(self.netMain)
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                        darknet.network_height(self.netMain),3)

        self.data_dir = os.path.expanduser(path+'/face_data')
        self.faces_folder_path = self.data_dir + '/users/'

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.data_dir + '/dlib/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1(self.data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')

        self.face_encodings, self.person_names = self.load_face_encodings()
        self.faceClassifier = cv2.CascadeClassifier(self.data_dir + '/dlib/haarcascade_frontalface_default.xml')
        #rn.recognize_faces_in_video(face_encodings, person_names)

        self.states = []
        self.skip = -1

    def convertBack(self, x, y, w, h):
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax


    def cvDrawBoxes(self, detections, img):
        for detection in detections:
            x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
            xmin, ymin, xmax, ymax = self.convertBack(
                float(x), float(y), float(w), float(h))
            pt1 = (xmin, ymin)
            pt2 = (xmax, ymax)
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
            cv2.putText(img,
                        detection[0].decode() +
                        " [" + str(round(detection[1] * 100, 2)) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        [0, 255, 0], 2)
        return img


    def get_face_encodings(self, face):
        bounds = self.face_detector(face, 1)
        faces_landmarks = [self.shape_predictor(face, face_bounds) for face_bounds in bounds]
        try:
            h = [np.array(self.face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]
        except:            
            return []
        return h


    def get_face_matches(self, known_faces, face):
        return np.linalg.norm(known_faces - face, axis=1)


    def find_match(self, known_faces, person_names, face):
        matches = self.get_face_matches(known_faces, face) # get a list of True/False
        min_index = matches.argmin()
        min_value = 1.0 - matches[min_index]
        if min_value > 0.45:
            return person_names[min_index]
        if min_value > 0.35:
            return person_names[min_index]
        return 'Not Found'


    def load_face_encodings(self):
        image_filenames = filter(lambda x: x.endswith('.jpg'), os.listdir(self.faces_folder_path))
        image_filenames = sorted(image_filenames)
        person_names = [x[:-4] for x in image_filenames]

        full_paths_to_images = [self.faces_folder_path + x for x in image_filenames]
        face_encodings = []

        win = dlib.image_window()

        for path_to_image in full_paths_to_images:
            print("Loading user: " + path_to_image)
            #face = io.imread(path_to_image)
            face = cv2.imread(path_to_image)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            faces_bounds = self.face_detector(face, 1)

            if len(faces_bounds) != 1:
                print("Expected one and only one face per image: " + path_to_image + " - it has " + str(len(faces_bounds)))
                exit()

            face_bounds = faces_bounds[0]
            face_landmarks = self.shape_predictor(face, face_bounds)
            face_encoding = np.array(self.face_recognition_model.compute_face_descriptor(face, face_landmarks, 1))

            win.clear_overlay()
            win.set_image(face)
            win.add_overlay(face_bounds)
            win.add_overlay(face_landmarks)
            face_encodings.append(face_encoding)

            #print(face_encoding)

            #dlib.hit_enter_to_continue()
        return face_encodings, person_names

    def detect_yolo(self, frame_read):
        prev_time = time.time()
        frame_resized = cv2.resize(frame_read,
                                   (darknet.network_width(rn.netMain),
                                    darknet.network_height(rn.netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(self.darknet_image, frame_rgb.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.35)
        #print(1/(time.time()-prev_time))
        
        return detections

    def detect(self, frame_read):
        draw_frame = frame_read.copy()
        gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
        overlay = frame_read.copy()
        cv2.rectangle(overlay, (0, 0), (640, 25), (0, 0, 0), -1)
        alpha = 0.8
        draw_frame = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)

        n_users = 0
        n_persons = 0

        current_states = []
        gain = 0.8
        self.skip += 1
        if self.skip % 4 % 2 == 0:
            # Yolo Detection
            detections = self.detect_yolo(frame_read.copy())
            filter_detections = []

            for detection in detections:
                if detection[0] == b'person': # It is a person
                    filter_detections.append(detection)

            if len(filter_detections) == 0: # Case Yolo didn't detected any person, try with dlib
                face_rects = self.faceClassifier.detectMultiScale( # Detect faces with dlib
                                    gray,
                                    scaleFactor = 1.1,
                                    minNeighbors = 5,
                                    minSize = (50, 50),
                                    flags = cv2.CASCADE_SCALE_IMAGE)

                if len(face_rects) > 0: # Case find any face
                    for (x, y, w, h) in face_rects:
                        face = draw_frame[y:y + h, x:x + w]
                        face_encodings_in_image = self.get_face_encodings(face)

                        if (face_encodings_in_image): 
                            match = self.find_match(self.face_encodings, self.person_names, face_encodings_in_image[0])
                            if match == "Not Found":
                                #cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                current_states.append(Person("Unknow", time.time(), -1, x, y, w, h, 0.8)) # Using DLIB
                            else:
                                #cv2.putText(draw_frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                n_users += 1
                                current_states.append(Person(match, time.time(), -1, x, y, w, h, 1.0)) # Using DLIB
                        else:
                            #cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            current_states.append(Person("Unknow", time.time(), -1, x, y, w, h, 0.8)) # Using DLIB

            else:
                for detection in filter_detections:
                    x1, y1, w1, h1 = detection[2][0],\
                        detection[2][1],\
                        detection[2][2],\
                        detection[2][3]
                    xmin, ymin, xmax, ymax = self.convertBack(
                        float(x1), float(y1), float(w1), float(h1))
                    sx = 640.0/self.darknet_width
                    sy = 360.0/self.darknet_height
                    xmin = int(xmin*sx)
                    ymin = int(ymin*sy)
                    xmax = int(xmax*sx)
                    ymax = int(ymax*sy)
                    pt1 = (xmin, ymin)
                    pt2 = (xmax, ymax)
                    cropped = gray[ymin:ymax, xmin:xmax]

                    face_rects = self.faceClassifier.detectMultiScale( # Detect faces with dlib
                                gray,
                                scaleFactor = 1.1,
                                minNeighbors = 5,
                                minSize = (50, 50),
                                flags = cv2.CASCADE_SCALE_IMAGE)

                    if len(face_rects) > 0:
                        for (x, y, w, h) in face_rects:
                            face = gray[y:y + h, x:x + w]
                            face_encodings_in_image = self.get_face_encodings(face)
                            #x += xmin
                            #y += ymin
                            if (face_encodings_in_image):
                                match = self.find_match(self.face_encodings, self.person_names, face_encodings_in_image[0])
                                if match == "Not Found":
                                    #cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                    #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                    current_states.append(Person("Unknow", time.time(), -1, x, y, w, h, 0.8)) # Using DLIB
                                else:
                                    #cv2.putText(draw_frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                    n_users += 1
                                    current_states.append(Person(match, time.time(), -1, x, y, w, h, 1.0)) # Using DLIB
                            else:
                                #cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                #cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                current_states.append(Person("Unknow", time.time(), -1, x, y, w, h, 0.8)) # Using DLIB
                    else:
                        #cv2.rectangle(draw_frame, pt1, pt2, (255, 0, 0), 1)
                        #cv2.putText(draw_frame, "Unknow", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        current_states.append(Person("Unknow", time.time(), -1, int(xmin), int(ymin), int((xmax-xmin)), int((ymax-ymin)), 0.8)) # Using YOLO

        if not self.states:
            print("Number of States: ", len(self.states))
            self.states = current_states
        else: # update
            old_states = []
            future_states = []
            visited_states = []
            print("Number of States: ", len(self.states), len(current_states))
            n_users = 0
            n_persons = 0
            for i in range(len(self.states)):
                state = self.states[i]
                state.t_to = time.time()
                state.prob = max(0.0, state.prob - 0.05)
                near_state = Person("Unknow", 0, 0, 0, 0, 0, 0, 0.0)
                for current_state in current_states:
                    if current_state.name == state.name and state.name != "Unknow":
                        print("Same name case!")
                        near_state = current_state
                        near_state.prob = 1.0 # prior
                        break

                    A = state.area()
                    B = current_state.area()
                    C = state.area_intersection(current_state)
                    relAB = min(B/A, A/B)
                    relAC = C/A
                    current_state.prob = (1-0.8)*relAB + (0.8)*relAC
                    print("Unknow State: ", current_state.prob, " ", near_state.prob)
                    if current_state.prob > near_state.prob and relAC > 0.5:
                        near_state = current_state

                if near_state.prob > 0.0: # position update
                    state.x = int(state.x*(1.0 - gain) + near_state.x*(gain))
                    state.y = int(state.y*(1.0 - gain) + near_state.y*(gain))
                    state.w = int(state.w*(1.0 - gain) + near_state.w*(gain))
                    state.h = int(state.h*(1.0 - gain) + near_state.h*(gain))
                    state.prob = near_state.prob
                    if state.name == "Unknow":
                        state.name = near_state.name

                self.states[i] = state
                if self.states[i].prob == 0.0:
                    old_states.append(self.states[i])
                    continue

                count = 0
                for m_state in self.states:
                    if state.x == m_state.x and state.y == m_state.y:
                        count += 1
                    if count > 1:
                        old_states.append(m_state)

                if state.name == "Unknow":
                    cv2.rectangle(overlay, (state.x, state.y), (state.x + state.w, state.y + 20), (0, 0, 0), -1)
                    alpha = 0.3
                    draw_frame = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)
                    cv2.rectangle(draw_frame, (state.x, state.y), (state.x + state.w, state.y + state.h), (0, 0, 255), 2)
                    cv2.putText(draw_frame, state.name + " | " + str(int(state.t_to-state.t_from))+" s", (state.x+5, state.y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    n_persons += 1
                else:
                    cv2.rectangle(overlay, (state.x, state.y), (state.x + state.w, state.y + 20), (0, 0, 0), -1)
                    alpha = 0.3
                    draw_frame = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)
                    cv2.rectangle(draw_frame, (state.x, state.y), (state.x + state.w, state.y + state.h), (0, 255, 0), 2)
                    cv2.putText(draw_frame, state.name + " | " + str(int(state.t_to-state.t_from))+" s", (state.x+5, state.y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    n_users += 1
                    n_persons += 1

            for current_state in current_states:
                matches = 0
                A = current_state.area()
                for state in self.states:
                    B = current_state.area_intersection(state)
                    if B/A > 0.3:
                        matches += 1
                if not matches:
                    future_states.append(current_state)

            for state in old_states: # remove old states
                try:
                    self.states.remove(state)
                except:
                    pass

            print("Future States: ", len(future_states))
            for future_state in future_states:
                states.append(future_state)

            print("==============================")
            for state in self.states:
                state.show()
                print("\n")

        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        cv2.putText(draw_frame, "InteliCam  |  "+ date_time +"  |  Users: " + str(n_users) + "  |  "+ \
                                            "Persons: " + str(n_persons) + "  |  Quit (q)",
                                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            [255, 255, 255], 1)
        cv2.imshow("Frame", draw_frame)
        key = cv2.waitKey(3) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            cv2.destroyAllWindows()
            return False, self.states, draw_frame

        return True, self.states, draw_frame

    # function to get the output layer names 
    # in the architecture
    def get_output_layers(self,net):
        
        layer_names = net.getLayerNames()
        
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers

    # function to draw bounding box on the detected object with class name
    def draw_bounding_box(self,img, class_id, confidence, x, y, x_plus_w, y_plus_h):


        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0, 0, 255), 2)

        #cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":

    # Start Yolo Setup
    rn = YOLO_NN(path='.', display=True)

    # initialize video input
    if len(sys.argv) > 1:
        cap = cv2.VideoCapture(int(sys.argv[1]))
    else:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    while True:
        ret, frame_read = cap.read()
        status, states, draw_frame = rn.detect(frame_read)

        if not status:
            break
        
        

        

    