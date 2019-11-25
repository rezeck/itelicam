#!/usr/bin/env python3.5
import os
import dlib
import numpy as np
#from skimage import io
import cv2


class YOLO_NN:
    def __init__(self, yoloDataFolder):
        
        WEIGHTS_FILE = str(yoloDataFolder) + "/yolov3.weights"
        CFG_FILE = str(yoloDataFolder) + "/yolov3.cfg"
        CLASSES_FILE = str(yoloDataFolder) + "/yolov3.txt"

        print("WEIGHTS_FILE: " + WEIGHTS_FILE)
        print("CFG_FILE: " + CFG_FILE)
        print("CLASSES_FILE: " + CLASSES_FILE)

        # read class names from text file
        self.classes = None
        with open(CLASSES_FILE, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(WEIGHTS_FILE, CFG_FILE)

        self.data_dir = os.path.expanduser('face_data')
        self.faces_folder_path = self.data_dir + '/users/'

        self.face_detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(self.data_dir + '/dlib/shape_predictor_68_face_landmarks.dat')
        self.face_recognition_model = dlib.face_recognition_model_v1(self.data_dir + '/dlib/dlib_face_recognition_resnet_model_v1.dat')


    def get_face_encodings(self, face):
        bounds = self.face_detector(face, 1)
        faces_landmarks = [self.shape_predictor(face, face_bounds) for face_bounds in bounds]
        return [np.array(self.face_recognition_model.compute_face_descriptor(face, face_pose, 1)) for face_pose in faces_landmarks]


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

            print(face_encoding)

            #dlib.hit_enter_to_continue()
        return face_encodings, person_names

    def detect(self, image):  
        Width = 640
        Height = 360
        scale = 0.00392 # 0.00392

        blob = cv2.dnn.blobFromImage(cv2.resize(image, (416,416)), scale, (416,416), (0,0,0), True, crop=False)

        # set input blob for the network
        self.net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.get_output_layers(self.net))
        
        # initialization
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        # for each detetion from each output layer 
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        
        new_boxes = map(lambda k: boxes[k[0]], indices)
        new_confidences = map(lambda k: confidences[k[0]],indices)
        new_classIds = map(lambda k: class_ids[k[0]],indices)
        
        return new_boxes, new_confidences, new_classIds

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
    # initialize video input
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 360)

    rn = YOLO_NN('yolov3')

    face_encodings, person_names = rn.load_face_encodings()
    faceClassifier = cv2.CascadeClassifier(rn.data_dir + '/dlib/haarcascade_frontalface_default.xml')
    #rn.recognize_faces_in_video(face_encodings, person_names)

    while True:
        ret, frame_read = cap.read()
        draw_frame = frame_read
        gray = cv2.cvtColor(frame_read, cv2.COLOR_BGR2GRAY)

        face_rects = faceClassifier.detectMultiScale(
            gray,
            scaleFactor = 1.1,
            minNeighbors = 5,
            minSize = (50, 50),
            flags = cv2.CASCADE_SCALE_IMAGE)
        if len(face_rects) > 0:
            for (x, y, w, h) in face_rects:
                face = draw_frame[y:y + h, x:x + w]
                face_encodings_in_image = rn.get_face_encodings(face)
                if (face_encodings_in_image):
                    match = rn.find_match(face_encodings, person_names, face_encodings_in_image[0])
                    if match == "Not Found":
                        cv2.putText(draw_frame, "Unknow", (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    else:
                        cv2.putText(draw_frame, match, (x+5, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.rectangle(draw_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            boxes, confidences, classIds = rn.detect(frame_read)
            rects = []
        
            # go through the detections remaining
            # after nms and draw bounding box
            for (box,classId) in zip(boxes,classIds):
                print(classId)
                if(classId==0):
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    cv2.putText(draw_frame, "Unknow", (round(x)+5, round(y)-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    rn.draw_bounding_box(frame_read, 0, 100, round(x), round(y), round(x+w), round(y+h))
    
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        cv2.imshow("Frame", frame_read)

    cv2.destroyAllWindows()