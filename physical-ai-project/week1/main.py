import cv2
import numpy as np
import time
from collections import deque

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠 연결 실패")
        return
    else:
        print("웹캠 연결 성공")
    
    # set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    start_time = time.time()
    fps_history = deque(maxlen=30)

    while True:
        ret,frame = cap.read()

        if not ret:
            break
        
        gray, blur, edges = preprocess(frame)

        current_time = time.time()
        avg_fps = calculate_fps(start_time, current_time, fps_history)
        start_time = current_time # update
        
        frame = draw_fps(frame, avg_fps)
        combined = combine_frames(frame, edges)

        cv2.imshow("original & edge image",combined)

        if cv2.waitKey(1)==ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def to_grayscale(frame):
    return cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

def apply_blur(gray,kernel_size):
    return cv2.GaussianBlur(gray,kernel_size,0)

def detect_edges(blur,sigma=0.33):
    '''
    Reference:
    Adrian Rosebrock, "Zero-parameter, automatic Canny edge detection
    with Python and OpenCV," PyImageSearch, April 6, 2015.
    
    Adapted from:
    https://pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    '''
    median = np.median(blur)
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(blur, lower, upper)

def preprocess(frame):
    gray = to_grayscale(frame)
    blur = apply_blur(gray,(1,1))
    edges = detect_edges(blur)
    return gray, blur, edges
    
def calculate_fps(start_time, end_time, fps_history):
    t = end_time - start_time
    
    if t > 0:
        fps = 1.0 / t
        fps_history.append(fps)
    
    if not fps_history:
        return 0.0
    
    return sum(fps_history) / len(fps_history)
    
def draw_fps(frame,fps):
    cv2.putText(
        img = frame,
        text = f"FPS: {fps:.1f}",
        org = (20,50),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.8,
        color = (0,255,0),
        thickness = 2
    )
    return frame

def combine_frames(frame,edges):
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined = np.hstack([frame, edges_colored])
    return combined

if __name__=="__main__":
    main()