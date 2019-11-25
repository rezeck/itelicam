import argparse
import asyncio
import logging
import os
import random
from threading import Thread, Lock


import cv2
from av import VideoFrame

from aiortc import (
    RTCIceCandidate,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder
from aiortc.contrib.signaling import ApprtcSignaling

ROOT = os.path.dirname(__file__)
PHOTO_PATH = os.path.join(ROOT, "loading.png")

global mutex
global im
im = None

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
            frame = VideoFrame.from_ndarray(im, format="bgr24")
            frame.pts = pts
            frame.time_base = time_base
                    
        return frame


async def run(pc, signaling):
    def add_tracks():
        pc.addTrack(VideoImageTrack())

    # connect to websocket and join
    params = await signaling.connect()

    if params["is_initiator"] == "true":
        # send offer
        add_tracks()
        await pc.setLocalDescription(await pc.createOffer())
        await signaling.send(pc.localDescription)

    # consume signaling
    while True:
        obj = await signaling.receive()

        if isinstance(obj, RTCSessionDescription):
            await pc.setRemoteDescription(obj)

            if obj.type == "offer":
                # send answer
                add_tracks()
                await pc.setLocalDescription(await pc.createAnswer())
                await signaling.send(pc.localDescription)
        elif isinstance(obj, RTCIceCandidate):
            pc.addIceCandidate(obj)
        elif obj is None:
            print("Exiting")
            break


def proc_image(name):
    global mutex
    global im

    logging.debug("[PROCESSING] Starting image capture")
    capture = cv2.VideoCapture(1)

    while(1):
        mutex.acquire()
        try:
            ret, im = capture.read()
        finally:
            mutex.release()
        
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AppRTC")
    parser.add_argument("room", nargs="?")
    parser.add_argument("--play-from", help="Read the media from a file and sent it."),
    parser.add_argument("--record-to", help="Write received media to a file."),
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    if not args.room:
        args.room = "".join([random.choice("0123456789") for x in range(10)])

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    # create signaling and peer connection
    signaling = ApprtcSignaling(args.room)
    pc = RTCPeerConnection()

    # Starting image processing
    mutex = Lock()
    proc_thread = Thread(target=proc_image, args=(0,))
    proc_thread.start()

    # run event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            run(pc=pc, signaling=signaling)
        )
    except KeyboardInterrupt:
        pass
    finally:
        # cleanup
        loop.run_until_complete(signaling.close())
        loop.run_until_complete(pc.close())