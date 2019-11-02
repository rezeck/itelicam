# import core features
from threading import Thread, Lock
from queue import Queue
from time import sleep
import cv2
from flask import Flask, render_template, Response, jsonify

from yolo_opencv import YOLO_NN

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 360
QUEUE = Queue(maxsize=5)
N_PERSONS = 0

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, DEFAULT_WIDTH)
        self.stream.set(4, DEFAULT_HEIGHT)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
			# if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
	
    def read(self):
        # return the frame most recently read
        return self.frame
 
    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

class DetectionVideoStream(WebcamVideoStream):
    def __init__(self):
        super().__init__()
    
    def update(self):
        global N_PERSONS,QUEUE
         # keep looping infinitely until the thread is stopped
        while True:
			# if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            boxes, confidences, classIds = DETECTOR_DNN.detect(self.frame)

            # go through the detections remaining
            # after nms and draw bounding box
            N_PERSONS = 0
            for (box,classId) in zip(boxes,classIds):
                if(classId==0):
                    N_PERSONS += 1
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]

                    DETECTOR_DNN.draw_bounding_box(self.frame, 0, 100, round(x), round(y), round(x+w), round(y+h))
            print('Inserting on queue')
            QUEUE.put(self.frame)


app = Flask(__name__, template_folder="html")

# initiate the neural netword
DETECTOR_DNN = YOLO_NN('yolo_opencv/yolov3');

detector = DetectionVideoStream()
detector.start()

def genVideo():
    while True:
        frame = QUEUE.get()
        (flag,encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue

        QUEUE.task_done()
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
         

@app.route('/')
def root():
    return render_template("flask.html")

@app.route("/stream")
def stream():
	print("Starting MJPEG Stream")
	return Response(genVideo(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/npersons')
def n_persons():
    resp = jsonify({ "n_persons":N_PERSONS })
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-Type'] = 'application/json'
    return resp

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8000", debug=True,
		threaded=True, use_reloader=False)

detector.stop()
