import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

latest_face = None

 #Callback 
def result_callback(result, output_image,timestamp_ms):
    global latest_face
    if result.face_landmarks:
        latest_face = result.face_landmarks[0]
        
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback
)

# THE LIVE STREAM
cap = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        timestamp = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)
        
        if latest_face:
            h,w,_ = frame.shape
            
            for point in latest_face:
                px = int(point.x * w)
                py = int(point.y * h)
                
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1 )
            
        cv2.imshow('Face Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()