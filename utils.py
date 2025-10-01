# ============================================== FUNCTIONS FOR FACE AND EMOTION RECOGNITON ==============================================#
import mediapipe as mp
import cv2
import numpy as np
from sklearn.preprocessing import normalize
import os
import tensorflow as tf

## Face detection model
def model_preparation():
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_det_model = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

    return face_det_model

## Face detection function
def face_detection(image, model, detection):
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = detection.process(rgb_frame)
    return results  

def get_face_roi(image, model):

    # Converte l'immagine PIL in formato OpenCV (BGR)
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Rilevamento del volto
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            roi = frame[y:y + h, x:x + w]  # Estrai la ROI del volto
            return roi
    return None    

######################################################## FUNCTIONS TO BE USED WHEN PERFORMING EMOTION RECOGNITION WITH THE MOBILENETV2 MODEL ########################################################
def deserialize_compound_loss(config, custom_objects=None):
    config.pop('name', None)
    return CompoundLoss(**config)

def fine_tuning(probs):
    probs[:, 1] = probs[:, 1] + probs[:, 4]

    # Switch of the "surprise"  and "neutral" classe (columns 4 and 5)
    probs[:, [4, 5]] = probs[:, [5, 4]]

    # Take just the first 4 classes
    probs = probs[:, :5]

    max_conf = np.max(probs, axis=1)
    predicted_classes = np.argmax(probs, axis=1) # contiene gli indici di max_conf

    # Force the model towards the class "Other" when the confidence scores are below the threshold
    predicted_classes[max_conf < 0.37] = 1
    return probs, predicted_classes