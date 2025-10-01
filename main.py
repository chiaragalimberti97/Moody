#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 10:41:11 2024

@author: roby
"""
from flask import Flask, render_template, redirect, url_for, session, request, Response, flash, jsonify
import cv2
import os
import numpy as np
import time
import base64
from time import time
from datetime import datetime
from io import BytesIO
from PIL import Image
import tensorflow as tf
from collections import Counter
from tensorflow.keras.models import load_model
from keras.saving import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.cache_handler import FlaskSessionCacheHandler
import random
from flask_cors import CORS
from deepface import DeepFace
from dotenv import load_dotenv
import os

# Functions from complementary files
from utils import model_preparation, face_detection, get_face_roi, deserialize_compound_loss, fine_tuning
import threading
import queue

load_dotenv() 
app = Flask(__name__)
CORS(app)


# Configuration to avoid probelm with cookies
app.config['SESSION_COOKIE_SAMESITE'] = None  
app.config['SESSION_COOKIE_SECURE'] = False   


# Spotify API configuration

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
redirect_uri = 'http://127.0.0.1:5000/callback'
scope = 'playlist-modify-public playlist-modify-private user-library-read user-read-private user-read-playback-state user-modify-playback-state streaming app-remote-control'


cache_handler = FlaskSessionCacheHandler(session)

sp_oauth = SpotifyOAuth(
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=redirect_uri,
    scope=scope,
    cache_handler=cache_handler,
    show_dialog=True  # Show login every time
)

sp = spotipy.Spotify(auth_manager=sp_oauth)


UPLOAD_FOLDER = 'images/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.secret_key = 'secret_key'  

camera=cv2.VideoCapture(0)
if not camera.isOpened():
       print("Error, impossible to access to camera")


# Load of the fine-tuned VGG16 model
classifier = load_model('best_vgg16_96_CCE_2.keras')
emotion_labels = ['Angry', 'Happy', 'Sad','Neutral', 'Other']


########################################## CODE TO USE THE MOBILENETV2 MODEL FOR EMOTION RECOGNITION ##########################################

# @tf.keras.utils.register_keras_serializable()
# class FisherLoss(tf.keras.losses.Loss):
#     def __init__(self, alpha=0.9, delta=0.01, num_classes=6): # changed num_classes to 6
#         super(FisherLoss, self).__init__()
#         self.alpha = alpha
#         self.delta = delta
#         self.mu = tf.Variable(tf.random.normal([num_classes, 6]), trainable=False)  # Mu is a variable, changed shape to [6,6]
#     def call(self, y_true, y_pred):
#         loss_1 = 0.0
#         batch_size = tf.shape(y_pred)[0]
#         predicted_class_index = tf.argmax(y_pred, axis=1)
#         gathered_mu = tf.gather(self.mu, predicted_class_index)
#         loss_1 = tf.reduce_sum(tf.reduce_mean(tf.square(y_pred - gathered_mu), axis=1)) / tf.cast(batch_size, tf.float32)
#         loss_2 = 0.0
#         num_classes = self.mu.shape[0]
#         for i in range(num_classes):
#             for j in range(i + 1, num_classes):
#                 loss_2 += tf.reduce_mean(tf.square(self.mu[i] - self.mu[j]))  # MSE
#         loss_2 = self.delta * loss_2 / (num_classes * (num_classes - 1) / 2)
#         loss = loss_1 - loss_2
#         indices = tf.cast(tf.argmax(y_true, axis=1), tf.int32)
#         updates = self.alpha * y_pred + (1 - self.alpha) * tf.gather(self.mu, indices)
#         self.mu.assign(tf.tensor_scatter_nd_update(self.mu, tf.expand_dims(indices, axis=1), updates))
#         return loss

# @tf.keras.utils.register_keras_serializable()
# class CompoundLoss(tf.keras.losses.Loss):
#     def __init__(self, theta=0.1):
#         super(CompoundLoss, self).__init__()
#         self.fisher_loss = FisherLoss(alpha=0.9, delta=0.01)
#         self.xentropy_loss = tf.keras.losses.CategoricalCrossentropy()
#         self.theta = theta
#     def call(self, y_true, y_pred):
#         return self.xentropy_loss(y_true, y_pred) + self.theta * self.fisher_loss(y_true, y_pred)   

# classifier = load_model('best_mobnet_96_Fisher_CCE.keras', custom_objects={'CompoundLoss': deserialize_compound_loss,
#                                    'FisherLoss': FisherLoss})

# emotion_labels = ['Angry', 'Other', 'Happy', 'Sad','Neutral']       

##################################################################################################################################################


# ============================================== HOMEPAGE ============================================== #
@app.route('/')
def home():
    return render_template('home.html')  


# ============================================== MENU PAGE ============================================== #
@app.route('/option')
def option():
    # Check that both enroll and spotify have been visited before allowing to click on the modality button
    if 'enroll' not in session or 'spotify' not in session: 
        start_disabled = True
    else:
        start_disabled = False
        
    return render_template('option.html', start_disabled=start_disabled)


# ============================================== ENROLLEMENT PAGE ==============================================#
@app.route('/enroll')
def enroll():
    session['enroll'] = True  
    return render_template('enroll.html')


# ============================================== NEW USER SAVE PAGE ==============================================#
@app.route('/save_enrollment', methods=['POST'])
def save_enrollment():
    photo_data = request.form['photo']
    
    # base64 image decodification
    image_data = base64.b64decode(photo_data.split(',')[1])
    image = Image.open(BytesIO(image_data))

    # Detect face and ROI
    face_det_model = model_preparation()
    face_roi = get_face_roi(image, face_det_model)

    if face_roi is not None:
        # Save ROI as a separate image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        roi_image_name = f'roi_user_{timestamp}.jpg'
        roi_image_path = os.path.join(app.config['UPLOAD_FOLDER'], roi_image_name)
        cv2.imwrite(roi_image_path, face_roi)  

    else:
        return None

    try:
        
        flash("Enrollment completed!")        
        return render_template('success.html', success_message = "Enrollment completed!")
    
    except Exception as e:
        flash(f"Error while saving the image: {str(e)}")
        return render_template('success.html', success_message = None)


# ============================================== PERSONALIZATION PAGE =============================================== #
## Personalization premium
@app.route('/personalization', methods=["GET", "POST"])
def personalization():

    # Load  and save preferences from the personalization page
    if request.method == "POST":
        happy_playlist = request.form.get("Happy")
        sad_playlist = request.form.get("Sad")
        angry_playlist = request.form.get("Angry")
        neutral_playlist = request.form.get("Neutral")

        return redirect(url_for('start'))  

    return render_template('personalization.html')

## Personalization naive
@app.route('/personalization_naive', methods=["GET", "POST"])
def personalization_naive():

    # Load and save preferences from the personalization page
    if request.method == "POST":
        happy_playlist = request.form.get("Happy")
        sad_playlist = request.form.get("Sad")
        angry_playlist = request.form.get("Angry")
        neutral_playlist = request.form.get("Neutral")
        return redirect(url_for('start_naive'))  

    return render_template('personalization_naive.html')  

# ============================================== MODE SELECTION PAGE ============================================== #
@app.route('/modality')
def modality():
    
    mode = request.args.get('mode', 'spotify_naive')  # Default on 'spotify_naive'
    session['mode'] = mode  
    return render_template('modality.html') 


# ================================= Face detection and emotion recogntion functions ==================================== #

## Threading for face verification
def verify_face(face_path, emb_file_path, result_queue):
    result = DeepFace.verify(
        img1_path=face_path,
        img2_path=emb_file_path,
        enforce_detection=False,
        model_name='SFace',
        distance_metric='cosine'
    )
    result_queue.put(result)
result_queue = queue.Queue()

## Real time face detection, face verification and emotion recognition
def face_emotion_recognition():
    
    emo=[]
    emo_prec=None
    face_det_model = model_preparation()
    cap = cv2.VideoCapture(0)
    counter = 0

    with face_det_model:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # Face detection
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_det_model.process(rgb_frame)

            if results.detections:
                counter +=1
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # ROI extraction
                    face_roi = frame[y:y + h, x:x + w]
                    if face_roi.any():
                        roi_color = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    
                    # ROI path
                    temp_face_path = 'temp_face.jpg'
                    cv2.imwrite(temp_face_path, roi_color)
                    if counter %7 ==0:
                        for known_users in os.listdir('images/'):
                            known_users_path = os.path.join('images/', known_users)
                            
                            if known_users.endswith('.jpg'):  
                                known_users_path = os.path.join('images/', known_users)

                                # Face verification
                                verification_thread = threading.Thread(
                                    target=verify_face,
                                    args=(temp_face_path, known_users_path, result_queue)
                                )
                                verification_thread.start()
                                verification_thread.join()

                                if not result_queue.empty():
                                    result = result_queue.get()

                                    # Authorization block
                                    if result['verified']:  
                                        roi_96 = cv2.resize(roi_color, (96, 96), interpolation=cv2.INTER_AREA) 
                                        roi = roi_96.astype('float') / 255.0
                                        roi = img_to_array(roi)
                                        roi = np.expand_dims(roi, axis=0)
                                        
                                        # Emotion recognition with VGG16
                                        prediction = classifier.predict(roi)[0]
                                        global label
                                        label = emotion_labels[prediction.argmax()]
                                        label_position = (x, y)
                                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        print(label)
                                        emo.append(label)

                                        ############## INSTRUCIONS TO USE THE MOBILENETV2 MODEL FOR EMOTION RECOGNITION ##############
                                        # preds = classifier.predict(roi)
                                        # probs, pred_classes = fine_tuning(preds)
                                        # print(probs)
                                        # print(pred_classes.item())
                                        # global label
                                        # label = emotion_labels[pred_classes.item()]
                                        # label_position = (x, y)
                                        # cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        # print(label)
                                        # emo.append(label)
                                        ##############################################################################################

                    if counter % 42 == 0:
                        freq = Counter(emo)
                        global main_emo 

                        # If at least one emotion has been detected, main_emo is going to be the most frequent
                        if len(freq) > 0:
                            temp = max(freq, key = freq.get)  
                            print("\n\nMain emo is:\n\n")
                            print(temp)
                            print("\n")   
                            if temp == "Other":
                                main_emo = emo_prec
                            else: main_emo = temp

                            # If the emotion hasn't changed, nothing will happen
                            if ( main_emo == emo_prec ):
                                print("\n\nNo switch in emotion\n\n")
                            else:  emo_prec=main_emo                                             
                            emo = []
                # Frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                frame = buffer.tobytes()
                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            else:
                print("No face found")    


# ============================================== START PAGE ============================================== #

## Premium start page
@app.route('/start', methods=["GET", "POST"])
def start():
    video_url = url_for('video_feed')  
    playlist_url = None
    main_emo = globals().get('main_emo', None)
    
    
    if request.method == "POST":

        # Load preferences about playlist
        happy_playlist = request.form.get("Happy")
        sad_playlist = request.form.get("Sad")
        angry_playlist = request.form.get("Angry")
        neutral_playlist = request.form.get("Neutral")
        session['playlists'] = {
            'Happy': happy_playlist,
            'Sad': sad_playlist,
            'Angry': angry_playlist,
            'Neutral': neutral_playlist
        }
    return render_template('start.html', video_url=video_url, main_emo = main_emo, playlist_url = playlist_url)


## Naive start page
@app.route('/start_naive', methods=["GET", "POST"])
def start_naive():
    video_url = url_for('video_feed') 
    playlist_url = None
    main_emo = globals().get('main_emo', None)

    
    if request.method == "POST":

        # Load preferences about playlists
        happy_playlist = request.form.get("Happy")
        sad_playlist = request.form.get("Sad")
        angry_playlist = request.form.get("Angry")
        neutral_playlist = request.form.get("Neutral")

        spotify_playlists = {
            'Happy': {
                'Feel Good Happy Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIgG2NEOhqsD7?utm_source=generator',
                'Happy Pop': 'https://open.spotify.com/embed/playlist/37i9dQZF1DWVlYsZJXqdym?utm_source=generator',
                'Rhythm Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIdTEeP5FUSaF?utm_source=generator',
                'Rock and Roll Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIf9QdS3bOrgZ?utm_source=generator',
                'Disney Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIYny1QYEZW0j?utm_source=generator',
            },
            'Sad': {
                'Melancholy Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EVKuMoAJjoTIw?utm_source=generator',
                'Sad Late Night Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIh4v230xvJvd?utm_source=generator',
                'Heartbroken Sad Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIf7xoQBl4aZ1?utm_source=generator',
                'Quiet Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIgQnNDX2DOQP?utm_source=generator',
                'Angry Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIgNZCaOGb0Mi?utm_source=generator',
            },
            'Angry': {
                'Rage Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIhuCNl2WSFYd?utm_source=generator&theme=0',
                'Mix Metal': 'https://open.spotify.com/embed/playlist/37i9dQZF1EQpgT26jgbgRI?utm_source=generator',
                'Mix Punk': 'https://open.spotify.com/embed/playlist/37i9dQZF1EQqlvxWrOgFZm?utm_source=generator',
                'Soft Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIcNUtFW3CJZc?utm_source=generator',
                'Classic Classical Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIghNBbh3wJEC?utm_source=generator',
            },
            'Neutral': {
                'Chill Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EVHGWrwldPRtj?utm_source=generator&theme=0',
                'Mix Jazz': 'https://open.spotify.com/embed/playlist/37i9dQZF1EQqA6klNdJvwx?utm_source=generator',
                'Relaxing Classical Mix': 'https://open.spotify.com/embed/playlist/37i9dQZF1EIcWaLKce2hIf?utm_source=generator',
                'Mix Rock': 'https://open.spotify.com/embed/playlist/37i9dQZF1EQpj7X7UK8OOF?utm_source=generator',
                'Hyper Focus Noise': 'https://open.spotify.com/embed/playlist/37i9dQZF1DX6iSJxWbeWLf?utm_source=generator',
            }
        }

        session['playlists'] = {
            'Happy': spotify_playlists['Happy'].get(happy_playlist, ''),
            'Sad': spotify_playlists['Sad'].get(sad_playlist, ''),
            'Angry': spotify_playlists['Angry'].get(angry_playlist, ''),
            'Neutral': spotify_playlists['Neutral'].get(neutral_playlist, '')
        }

    
    return render_template('start_naive.html', video_url=video_url, main_emo = main_emo, playlist_url = playlist_url)


# ============================================== VIDEO HANDLING ============================================== #
@app.route('/video_feed')
def video_feed():
    # Part that handle real time streaming
    return Response(face_emotion_recognition(), mimetype='multipart/x-mixed-replace; boundary=frame')


#============================================== PLAYLIST MANAGEMENT ============================================== #

## Premium Playlist
@app.route('/get_playlist')
def get_playlist():
    # Load tha recognised emotion
    main_emo = globals().get('main_emo', None)
    
    # Load the preferences about playlists
    playlists = session.get('playlists', {})
    playlist_url = None

    # Handle spotify login issues 
    if main_emo and playlists.get(main_emo):
        token_info = session.get('token_info')
        if not session.get('token_info'):
            return redirect(url_for('login'))
        else: 
            token_info=session.get('token_info')
            sp = spotipy.Spotify(auth=token_info['access_token'])

        # Search in spotify the string playlist[main_emo] so the playlist associated to the detected emotion
        playlist_url = search_playlist(playlists[main_emo],token_info)

    if main_emo:
        print("\n\nMain emo is:")
        print({main_emo})
        print({playlists[main_emo]})
        print("\n\n")
    # Communicate with the html page the url of the playlist that have to be played
    return jsonify({'playlist_url': playlist_url, 'main_emo' : main_emo})

## Naive Playlist
@app.route('/get_playlist_naive')
def get_playlist_naive():
    # Load the detected emotion
    main_emo = globals().get('main_emo', None)
    
    # Load the preferences
    playlists = session.get('playlists', {})
    playlist_url = None

    if main_emo and playlists.get(main_emo):
        # Take the playlist associated to the detected emotion
        playlist_url = playlists[main_emo]

    # Communicate with the html page the url of the playlist that have to be played
    return jsonify({'playlist_url': playlist_url, 'main_emo': main_emo})


# ============================================== "BACK TO" BUTTONS ============================================== #
@app.route('/back_to_home')
def back_to_home():
    return redirect(url_for('home')) 

@app.route('/back_to_option')
def back_to_option():
    return redirect(url_for('option'))  

@app.route('/back_to_mode')
def back_to_mode():
    return redirect(url_for('modality'))      

# Cased on the chosen modality it redirects the user to the associated personalization page
@app.route('/back_to_option_2', methods=['POST'])
def back_to_option_2():
    mode = request.form['mode']  
    if mode == 'spotify_naive':
        return redirect(url_for('personalization_naive')) 
    elif mode == 'spotify_premium':
        return redirect(url_for('personalization')) 
    return redirect(url_for('option'))    


# ============================================== SPOTIFY TOKEN HANDLING ============================================== #

# Function to call to get the token info (handle also refreshing)
@app.route('/get_spotify_token')
def get_spotify_token():
    token_info = session.get('token_info')
    if not token_info:
        return None
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info
    token = token_info['access_token']

    return jsonify(token=token)


# ============================================== SEARCH RANDOM TRACK IN THE PLAYLIST ==============================================#
@app.route('/search_playlist')
def search_playlist(playlist,token_info):
    
    # if the token is not present redirect to login
    if not session.get('token_info'):
       return redirect(url_for('login'))
    # if its present it loads it 
    token_info =session.get('token_info')
    if sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
    print(token_info)

    # create an authenticated spotify client
    sp = spotipy.Spotify(auth=token_info['access_token'])
    print(playlist)

    playlist_name = playlist
    
    if not playlist_name:
        return "Please provide a playlist name.", 400

    # search the string playlist_name between spotify's playlists
    results = sp.search(q=f'playlist:{playlist_name}', type='playlist', limit=5)
    
    playlists = results.get('playlists', {}).get('items', [])
    playlists = [pl for pl in playlists if pl is not None]

    if not playlists:
        return f"No playlist found with the name '{playlist_name}'"
    print(results)
    print(playlists)

    #save playlist id and playlist name
    playlist_id = playlists[0]['id']
    playlist_name = playlists[0]['name']
    print(f"Playlist Name: {playlist_name}")
    
    # extract the track from the playlist
    tracks = sp.playlist_tracks(playlist_id).get('items', [])
    if not tracks:
        return "The playlist is empty."

    # pick a random track: this step is necessary otherwise each time the user is happy the playlist
    # would start with the same song

    random_track = random.choice(tracks)

    #save the track uri ( which uniquely identify the track)
    track_uri = random_track['track']['uri']
    
    track_name = random_track.get('track', {}).get('name', "Unknown Track")
    print(track_name)
   
    return track_uri


# ============================================== ELEMENTS FOR LOGIN AND ACCESS TO SPOTIFY ==============================================#
@app.route('/login', methods=['GET'])
def login():
    auth_url = sp_oauth.get_authorize_url()
    print(auth_url)
    return redirect(auth_url)

@app.route('/spotify')
def spotify():
    session['spotify'] = True
    return redirect(url_for('login'))

@app.route('/SuccessLogin')
def SuccessLogin():
    return render_template('spotify.html')

@app.route('/callback')
def callback():
    code = request.args.get('code')
    token_info = sp_oauth.get_access_token(code)
    session['token_info'] = token_info
    return redirect(url_for('SuccessLogin'))

# Refresh the spotify token before every incoming request
@app.before_request
def refresh_token():
    token_info = session.get('token_info', {})
    if token_info and sp_oauth.is_token_expired(token_info):
        token_info = sp_oauth.refresh_access_token(token_info['refresh_token'])
        session['token_info'] = token_info   
    print("Route refresh token")


if __name__ == '__main__':
    app.run(debug=True)