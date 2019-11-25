# internal imports
from tornado.gen import coroutine
from tornado.web import Application, RequestHandler
from tornado.ioloop import IOLoop
from queue import Queue

# local imports
from vprocess import DetectionVideoStream

QUEUE = Queue(maxsize=5)

detector = DetectionVideoStream()
detector.start()

class VideoStream(RequestHandler):
    @coroutine
    def get(self):
        ioloop = tornado.ioloop.IOLoop.current()

        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Connection', 'close')
        self.set_header('Content-Type', 'multipart/x-mixed-replace;boundary=--boundarydonotcross')
        self.set_header('Pragma', 'no-cache')


        self.served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:
            img = QUEUE.get()
            interval = 1.0
            if self.served_image_timestamp + interval < time.time():
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(str(img))
                self.served_image_timestamp = time.time()
                yield tornado.gen.Task(self.flush)
            else:
                yield tornado.gen.Task(ioloop.add_timeout, ioloop.time() + interval)
            
            QUEUE.task_done()

def make_app():
    urls = [("/video", VideoStream)]
    return Application(urls)
  
if __name__ == '__main__':
    app = make_app()
    app.listen(3041)
    try:
        tornado.ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
        tornado.ioloop.IOLoop.instance().stop()