import os
import cv2
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

latest_face = None
expression = 'None'
latest_blendshapes = {}

#EXPRESSION's 
Memes = {
    'default': 'MEMES/default.jpeg',
    'smiling': 'MEMES/smile.jpeg',
    'shocked': 'MEMES/shocked_dog.jpeg',
    'speed_shocked': 'MEMES/shocked_speed.jpeg',
    'speed_smile': 'MEMES/speed_smile.jpeg'
}

memes_paths = {}
for name, path in Memes.items():
    img = cv2.imread(path)
    if img is not None:
        memes_paths[name] = cv2.resize(img, (300, 300))
    else:
        print(f"[warn] could not load meme: {path}")


#CALLBACK
def result_callback(result, output_image, timestamp_ms):
    global latest_face, expression, latest_blendshapes
    
    if not result.face_landmarks:
        latest_face = None
        latest_blendshapes = {}
        expression = 'No Face Detected'
        return
    
    latest_face = result.face_landmarks[0]
    
    if result.face_blendshapes:
        bs_list = result.face_blendshapes[0]
        latest_blendshapes = {bs.category_name: bs.score for bs in bs_list}
        expression = classify_expression(latest_blendshapes)
        
def classify_expression(bs):
    
    eye_wide = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
    jaw_open = bs.get('jawOpen', 0)
    blink = (bs.get('eyeBlinkLeft', 0) + bs.get('eyeBlinkRight', 0)) / 2
    tongue= bs.get('tongueOut', 0)
    
    def is_smiling(bs):
        smile  = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
        return smile > 0.5 
    def is_shocked(bs):
        return eye_wide > 0.5
    def is_speed_shocked(bs):
        return eye_wide > 0.5 and jaw_open < 0.3
    def is_speed_smile(bs):
        return blink > 0.6 and smile > 0.5
    def is_tongue_out(bs):
        return tongue > 0.5

EXPRESSION_CHECKS = [
    ('smiling',  is_smiling),
    ('shocked', is_shocked),
    ('speed_shocked', is_speed_shocked),
    ('speed_smile', is_speed_smile),
]

def classify_expression(bs):
    for label, check_fn in EXPRESSION_CHECKS:
        if check_fn(bs):
            return label
    return 'neutral'
    
def overlay_meme(frame, meme_img, x=10, y=10):
    mh, mw = meme_img.shape[:2]
    fh, fw = frame.shape[:2]
    x2 = min(x + mw, fw)
    y2 = min(y + mh, fh)
    meme_crop = meme_img[0:y2-y, 0:x2-x]

    frame[y:y2, x:x2] = meme_crop
    return frame



base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback,
    output_face_blendshapes=True,
    num_faces=1
)

cap = cv2.VideoCapture(0)
with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        timestamp = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp)
        
        if latest_face:
            for point in latest_face:
                px= int(point.x * w)
                py= int(point.y * h)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
                
        meme_img = memes_paths.get(expression)
        if meme_img is not None:
            meme_x = w - meme_img.shape[1] - 10
            overlay_meme(frame, meme_img, x=meme_x, y=10)

        cv2.putText(frame, f'Expression: {expression}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 200, 255), 2)
        
        if latest_blendshapes:
            top = sorted(latest_blendshapes.items(), key=lambda x: -x[1])[:4]
            for i, (name, score) in enumerate(top):
                cv2.putText(frame, f'{name}: {score:.2f}',
                            (10, 60 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
         
        cv2.imshow('DumbMemes', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()