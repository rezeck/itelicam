import argparse
import asyncio
import logging
import os
import json
import random
from threading import Thread, Lock

import sys

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
    global mutex, im

    print("[PROCESSING] Starting image capture")
    capture = cv2.VideoCapture("test.mp4")
    capture.set(3, 720)
    capture.set(4, 1280)

    while(1):
        mutex.acquire()
        try:
            ret, im = capture.read()
            
            if ret:
                #cv2.imshow("Video", im)
                cv2.waitKey(30)
                pass
            else:
               print('no video')
               capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
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