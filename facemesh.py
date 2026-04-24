import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2

# 1. Configuration: Telling the "Task" how to behave
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, 
    num_faces=1
)

# 2. Create the Landmarker
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    
    # Load the image using OpenCV
    raw_image = cv2.imread("Face.jpeg")
    
    # MediaPipe needs its own Image format
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=raw_image)

    # 3. RUN THE AI
    result = landmarker.detect(mp_image)

    # 4. Interpret the Results
    if result.face_landmarks:
        # result.face_landmarks is a list (for multiple faces). 
        # We grab the first face detected [0].
        face = result.face_landmarks[0]
        h, w, _ = raw_image.shape

        # Let's target the corners of left eyes
        eye_left = [33, 133] 
        nose = [1] # Nose tip
        eye_right = [362, 263] # Corners of the right eye
        lips = [61, 291] # Corners of the lips
        

        for idx in eye_left:
            point = face[idx]
            px = int(point.x * w)
            py = int(point.y * h)
            
            # Draw a smaller blue dot for the eyes
            cv2.circle(raw_image, (px, py), 4, (255, 0, 0), -1)
        
        for i in eye_right:
            point=face[i]
            px=int(point.x*w)
            py=int(point.y*h)
            
            cv2.circle(raw_image, (px, py), 4, (255, 0, 0), -1 )
            
        for idx in nose:
            point = face[idx]
            px = int(point.x * w)
            py = int(point.y * h)
            
            # Draw a smaller red dot for the nose
            cv2.circle(raw_image, (px, py), 4, (0, 0, 255), -1)
            
        for i in lips:
            point=face[i]
            px=int(point.x*w)
            py=int(point.y*h)
            
            cv2.circle(raw_image, (px, py), 4, (0, 255, 0), -1 )
        
    # Display result
    cv2.imshow("MediaPipe Tasks Test", raw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()