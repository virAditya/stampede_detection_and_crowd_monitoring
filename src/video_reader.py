"""
Asynchronous Video Reader Module
"""
import threading
import queue


class AsyncVideoReader(threading.Thread):
    """Asynchronous video frame reader in separate thread"""
    def __init__(self, video_path, frame_buffer_size=30):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_buffer = queue.Queue(maxsize=frame_buffer_size)
        self.stop_flag = threading.Event()
        self.cap = None
        
    def run(self):
        import cv2
        self.cap = cv2.VideoCapture(self.video_path)
        
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            try:
                self.frame_buffer.put(frame, timeout=0.5)
            except queue.Full:
                pass
        
        self.cap.release()
    
    def get_frame(self, timeout=1.0):
        try:
            return self.frame_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2.0)
