import argparse
import asyncio
import logging
import os
import json
import random
from threading import Thread, Lock

import sys

from darknet import darknet

import cv2

from av import VideoFrame
from aiohttp import web

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)

ROOT = os.path.dirname(__file__)
PHOTO_PATH = os.path.join(ROOT, "loading.png")

mutex = None
im = None

netMain = None
metaMain = None
altNames = None

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
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

def load_yolo():
    global metaMain, netMain, altNames
    configPath = "./darknet/cfg/yolov3-tiny.cfg"
    weightPath = "./darknet/yolov3-tiny.weights"
    metaPath = "./darknet/cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
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
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass




class VideoImageTrack(VideoStreamTrack):
    """
    A video stream track that returns a rotating image.
    """

    def __init__(self):
        super().__init__()  # don't forget this!
        self.img = cv2.imread(PHOTO_PATH, cv2.IMREAD_COLOR)

    async def recv(self):
        global im
        pts, time_base = await self.next_timestamp()

        # create video frame
        if im == None:
            frame = VideoFrame.from_ndarray(self.img, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
        else:
            options = {"framerate": "30", "video_size": "640x480"}
            mutex.acquire()
            frame = VideoFrame.from_ndarray(im, format="bgr24")
            frame.pts = pts
            #frame.options = options
            frame.time_base = time_base
            mutex.release()
                    
        return frame


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    
    channel = pc.createDataChannel("data")
    async def send_pings():
        while True:
            channel.send("Hi")
            await asyncio.sleep(1)

    @channel.on("open")
    def on_open():
        asyncio.ensure_future(send_pings())

    @channel.on('message')
    def on_message(message):
        print("Receive a message: ", message)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange():
        print("ICE connection state is %s" % pc.iceConnectionState)
        if pc.iceConnectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    await pc.setRemoteDescription(offer)
    #for t in pc.getTransceivers():
    pc.addTrack(VideoImageTrack())
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


def proc_image(name):
    global metaMain, netMain, altNames
    global mutex, im

     # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

    logging.debug("[PROCESSING] Starting image capture")
    capture = cv2.VideoCapture(0)
    capture.set(3, 720)
    capture.set(4, 1280)

    while(1):
        ret, frame = capture.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(netMain),
                                darknet.network_height(netMain)),
                               interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        mutex.acquire()
        try:
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #im = cv2.resize(im, (320,280), interpolation=cv2.INTER_LINEAR)
        finally:
            mutex.release()
    im = None
    capture.release()
    cv2.destroyAllWindows()






async def index(request):
    content = open(os.path.join(ROOT, "index.html"), "r").read()
    return web.Response(content_type="text/html", text=content)


async def javascript(request):
    content = open(os.path.join(ROOT, "client.js"), "r").read()
    return web.Response(content_type="application/javascript", text=content)


pcs = set()

async def on_shutdown(app):
    # close peer connections
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yolo WebRTC")
    parser.add_argument('--port', type=int, default=8080,
        help='Port for HTTP server (default: 8080)')
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    load_yolo()

    # Starting image processing
    mutex = Lock()
    proc_thread = Thread(target=proc_image, args=(0,))
    proc_thread.start()

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get('/', index)
    app.router.add_get('/client.js', javascript)
    app.router.add_post('/offer', offer)
    web.run_app(app, port=args.port, host='127.0.0.1')

    # run event loop
    #loop = asyncio.get_event_loop()
    #try:
    #    loop.run_until_complete(
    #        run(pc=pc, signaling=signaling)
    #    )
    #except KeyboardInterrupt:
    #    pass
    #finally:
    #    # cleanup
    #    loop.run_until_complete(signaling.close())
    #    loop.run_until_complete(pc.close())