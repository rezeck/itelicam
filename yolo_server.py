# import core features
from threading import Thread, Lock
from queue import Queue
from time import sleep
from flask import Flask, render_template, Response, jsonify
import cv2

# import local modules
from vprocess import DetectionVideoStream

app = Flask(__name__)

detector = DetectionVideoStream()
detector.start()

def genVideo():
    while True:
        frame = detector.frame
        (flag,encodedImage) = cv2.imencode(".jpg", frame)

        if not flag:
            continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
         

@app.route('/')
def root():
    return jsonify({ "hello":"World" })

@app.route("/video")
def stream():
	print("Starting MJPEG Stream")
	return Response(genVideo(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    app.run(host="127.0.0.1", port="8000", debug=True,
		threaded=True, use_reloader=False)

detector.stop()
