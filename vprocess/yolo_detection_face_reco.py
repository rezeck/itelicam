#!/usr/bin/env python3.5
import os
import dlib
import numpy as np
import cv2
import time
import darknet
from ctypes import *
import math
import random


class YOLO_NN:
    def __init__(self, yoloDataFolder):

        self.configPath = yoloDataFolder + "/cfg/yolov3-tiny.cfg"
        self.weightPath = yoloDataFolder + "/yolov3-tiny.weights"
        self.metaPath = yoloDataFolder + "/cfg/coco.data"

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
        self.darknet_image = darknet.make_image(darknet.network_width(self.netMain),
                                        darknet.network_height(self.netMain),3)

        self.data_dir = os.path.expanduser(yoloDataFolder+'/face_data')
        self.faces_folder_path = self.data_dir + '/users/'

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.data_dir + '/dlib/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1(self.data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')

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
        min_value = matches[min_index]
        if min_value < 0.55:
            return person_names[min_index]+"! ({0:.2f})".format(min_value)
        if min_value < 0.58:
            return person_names[min_index]+" ({0:.2f})".format(min_value)
        if min_value < 0.65:
            return person_names[min_index]+"?"+" ({0:.2f})".format(min_value)
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

    def detect(self, frame_read):
        prev_time = time.time()
        frame_resized = cv2.resize(frame_read,
                                   (darknet.network_width(rn.netMain),
                                    darknet.network_height(rn.netMain)),
                                   interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        darknet.copy_image_from_bytes(self.darknet_image, frame_rgb.tobytes())

        detections = darknet.detect_image(self.netMain, self.metaMain, self.darknet_image, thresh=0.25)
        #print(1/(time.time()-prev_time))
        
        return detections

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
    rn = YOLO_NN('.')

    # initialize video input
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    face_encodings, person_names = rn.load_face_encodings()
    faceClassifier = cv2.CascadeClassifier(rn.data_dir + '/dlib/haarcascade_frontalface_default.xml')
    #rn.recognize_faces_in_video(face_encodings, person_names)

    while True:
        ret, frame_read = cap.read()
        
        draw_frame = frame_read.copy()
        gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)
        overlay = frame_read.copy()
        cv2.rectangle(overlay, (0, 0), (640, 35), (0, 0, 0), -1)
        alpha = 0.8
        draw_frame = cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0)

        # Yolo Detection
        detections = rn.detect(frame_read.copy())
        filter_detections = []

        n_users = 0
        n_persons = 0

        for detection in detections:
            if detection[0] == b'person': # It is a person
                filter_detections.append(detection)

        if len(filter_detections) == 0: # Case Yolo didn't detected any person, try with dlib
            face_rects = faceClassifier.detectMultiScale( # Detect faces with dlib
                                gray,
                                scaleFactor = 1.1,
                                minNeighbors = 5,
                                minSize = (50, 50),
                                flags = cv2.CASCADE_SCALE_IMAGE)

            n_persons = len(face_rects)

            if len(face_rects) > 0: # Case find any face
                for (x, y, w, h) in face_rects:
                    face = draw_frame[y:y + h, x:x + w]
                    face_encodings_in_image = rn.get_face_encodings(face)

                    if (face_encodings_in_image): 
                        match = rn.find_match(face_encodings, person_names, face_encodings_in_image[0])
                        if match == "Not Found":
                            cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        else:
                            cv2.putText(draw_frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            n_users += 1
                    else:
                        cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        else:
            for detection in filter_detections:
                x1, y1, w1, h1 = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = rn.convertBack(
                    float(x1), float(y1), float(w1), float(h1))
                sx = 640.0/416.0
                sy = 360.0/416.0
                xmin = int(xmin*sx)
                ymin = int(ymin*sy)
                xmax = int(xmax*sx)
                ymax = int(ymax*sy)
                pt1 = (xmin, ymin)
                pt2 = (xmax, ymax)
                cropped = gray[ymin:ymax, xmin:xmax]

                face_rects = faceClassifier.detectMultiScale( # Detect faces with dlib
                            gray,
                            scaleFactor = 1.1,
                            minNeighbors = 5,
                            minSize = (50, 50),
                            flags = cv2.CASCADE_SCALE_IMAGE)
                n_persons += 1 

                if len(face_rects) > 0:
                    for (x, y, w, h) in face_rects:
                        face = cropped[y:y + h, x:x + w]
                        face_encodings_in_image = rn.get_face_encodings(face)
                        #x += xmin
                        #y += ymin
                        if (face_encodings_in_image):
                            match = rn.find_match(face_encodings, person_names, face_encodings_in_image[0])
                            if match == "Not Found":
                                cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            else:
                                cv2.putText(draw_frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                n_users += 1
                        else:
                            cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                else:
                    cv2.rectangle(draw_frame, pt1, pt2, (0, 0, 255), 2)
                    cv2.putText(draw_frame, "Unknow", (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        cv2.putText(draw_frame, "InteliCam       Users: " + str(n_users) + "   |   "+ \
                                            "Persons: " + str(n_persons),
                                            (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                            [255, 255, 255], 1)
        cv2.imshow("Frame", draw_frame)
        key = cv2.waitKey(3) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        

    cv2.destroyAllWindows()