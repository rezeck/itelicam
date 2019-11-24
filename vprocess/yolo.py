import cv2
import numpy as np


class YOLO_NN:
    def __init__(self, yoloDataFolder):
        
        WEIGHTS_FILE = f'{yoloDataFolder}/yolov3.weights'
        CFG_FILE = f'{yoloDataFolder}/yolov3.cfg'
        CLASSES_FILE =f'{yoloDataFolder}/yolov3.txt'

        # read class names from text file
        self.classes = None
        with open(CLASSES_FILE, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # generate different colors for different classes 
        COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        # read pre-trained model and config file
        self.net = cv2.dnn.readNet(WEIGHTS_FILE, CFG_FILE)

    def detect(self,image):  
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


        cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), (0, 255, 0), 2)

        #cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__":
    # initialize video input
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 360)

    rn = YOLO_NN('yolov3')

    while True:
        ret, frame_read = cap.read()
        boxes, confidences, classIds = rn.detect(frame_read)

        # go through the detections remaining
        # after nms and draw bounding box
        for (box,classId) in zip(boxes,classIds):
            print(classId)
            if(classId==0):
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                rn.draw_bounding_box(frame_read, 0, 100, round(x), round(y), round(x+w), round(y+h))
        
        key = cv2.waitKey(1) & 0xFF
    
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        cv2.imshow("Frame", frame_read)

    cv2.destroyAllWindows()