import keras
from keras.models import load_model
import cv2
from deepface import DeepFace

# Runtime Evaluation Libraries
import time
import datetime

import os
import ffmpeg
import json
import numpy as np
import random

# Import of the Blink and Gaze Detection Module
import sys
base_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_path, 'gaze-tracking/'))
import module as m

# Models folder
base_path = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(base_path, 'models')

emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
model = load_model(os.path.join(models_folder, 'model_v6_23.hdf5'))
faceCascade = cv2.CascadeClassifier(os.path.join(models_folder, 'haarcascade_frontalface_default.xml'))
eyeCascade = cv2.CascadeClassifier(os.path.join(models_folder, 'haarcascade_eye_tree_eyeglasse.xml'))
smileCascade = cv2.CascadeClassifier(os.path.join(models_folder, 'haarcascade_smile.xml'))

def evaluacion_video(person_folder, person_id, video_url, duration):
  print('***Video analysis***')
  # Use the JSON with the video metadata to detect if it should be rotated
  metadata_json = os.path.join(person_folder, person_id + '_metadata.json')
  inicio_ev_video = time.time()
  metricas_video = os.path.join(person_folder, person_id + '_faces_smiles.json')
  metricas_parpadeos = os.path.join(person_folder, person_id +  '_blinking.json')
  metricas_emociones = os.path.join(person_folder, person_id + '_emotions.json')
  if os.path.exists(metricas_video)==False:
    video = cv2.VideoCapture(video_url)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    f = video.get(cv2.CAP_PROP_FPS)
    counter = 0
    closed_eyes_frame = 3
    total_blinks = 0
    parpadeos_parte_1 = 0
    parpadeos_parte_2 = 0
    frames_con_rostro = 0
    inicio_parte_1=0
    fin_parte_1 = (frame_count-1)/2
    if type(fin_parte_1)==float:
        fin_parte_1 = round(fin_parte_1, 0)
    inicio_parte_2 = fin_parte_1+1
    fin_parte_2 = frame_count-1
    emociones = []
    detection_output=[]
    rotation_degrees=0
    # Detect if video should be rotated
    with open(metadata_json) as json_file:  
        metadata = json.load(json_file)
    try:
      rotation_degrees = [stream for stream in metadata["streams"] if stream["codec_type"] == "video"][0]['side_data_list'][0]['rotation']
    except:
      pass

    # Reading video
    
    count=0
    frames_p_seg = round(video.get(cv2.CAP_PROP_FPS),0)
    frames_con_rostro = 0

    width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

    center = (width / 2, height / 2)

    # Evaluating one frame per second
    # For the frames where the measures will be obtained
    # Inicio segundo starts at -frames_p_seg as this increments according to the number of frames p sec of the video
    inicio_segundo = frames_p_seg*-1
    # final_segundo starts at -1 as it increases according to the number of frames p sec of the video
    final_segundo = -1
    # It indicates what the frame number is in the second. The maximum value is the frame rate.
    #frame_en_el_segundo = 0
    # Get the alteatory positions of the frames to be evaluated and store them in a list
    frames_a_evaluar = []
    for second in range(int(duration)):
      # inicio_segundo indicates the first frame of the second in question
      inicio_segundo+=frames_p_seg
      # final_segundo indicates the last frame of the second in question
      final_segundo+=frames_p_seg
      frames_a_evaluar.append(random.randint(inicio_segundo, final_segundo))
      
    
    for fae in frames_a_evaluar:
      try:
        if fae >= 0 & fae <= frame_count:
          video.set(cv2.CAP_PROP_POS_FRAMES,fae)
        while True:
           
                ret, frame = video.read() 
                
                frame_details={}
                frame_details["frame"]="frame%d"%fae
                faces_list = []
                
                smiles_list = []
                frames_for_video = []
                gaze_label = 'Unknown'
                blinking = False

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                

                if rotation_degrees == 90 or rotation_degrees==-90:  
                  frame = cv2.rotate(frame, cv2.ROTATE_180)
                  gray = cv2.rotate(gray, cv2.ROTATE_180)
                
                ### Find areas with faces using Haar cascade classifier
                faces = faceCascade.detectMultiScale(image= gray, scaleFactor= 1.1, minNeighbors= 4)
                # scaleFactor: how much the image size is reduced at each image scale
                # minNeighbors: how many neighbors each candidate rectangle should have to retain it

                # x, y coordinates, w (weight) and h (height) of each "face" rectangle in frame
                
                try:
                  analyze = DeepFace.analyze(frame, actions = ['emotion'], prog_bar=False)
                  emociones.append(analyze)
                except:
                  pass


                for (x, y, w, h) in faces:
                    eyes_list = []
                   
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    # Get the region of interest: face rectangle sub-image in gray and colored
                    faceROIGray = gray[y: y+h, x: x+w]
                    faceROIColored = frame[y: y+h, x: x+w]
                    


                    ### Find areas with eyes in faces using Haar cascade classifier
                    eyes = eyeCascade.detectMultiScale(faceROIGray)
                    
                    width = np.size(faceROIColored, 1)
                    height = np.size(faceROIColored, 0)
                    # x, y coordinates, w (weight) and h (height) of each "eye" rectangle in a face
                    for (ex, ey, ew, eh) in eyes:
                        if ey > height / 2:
                          
                          pass
                        else:
                            eyecenter = ex + ew / 2  # get the eye center
                            cv2.rectangle(faceROIColored, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                            eyes_list.append({"y_min":int(ey),"y_max":int(ey+eh),"x_min":int(ex),"x_max":int(ex+ew)})
                    
                    
                    if len(eyes_list)==1 or len(eyes_list)==2:
                        faces_list.append({"y_min":int(y),"y_max":int(y+h),"x_min":int(x),"x_max":int(x+w)})
                        # Detecting gaze
                        image, face = m.faceDetector(frame, gray)
                        if face is not None:
                          try:
                            image, PointList = m.faceLandmakDetector(frame, gray, face, False)
                            
                            
                            RightEyePoint = PointList[36:42]
                            LeftEyePoint = PointList[42:48]
                            leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
                            rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)
                            blinkRatio = (leftRatio + rightRatio)/2


                            if blinkRatio > 4:
                              counter += 1
                            else:
                              if counter > 0:
                                total_blinks+=1
                                counter = 0
                                
                            

                            mask, pos, color = m.EyeTracking(frame, gray, RightEyePoint)
                            maskleft, leftPos, leftColor = m.EyeTracking(frame, gray, LeftEyePoint)
                            gaze_label = pos
                           

                          except Exception as e:
                            print(e)
                            pass
   
                    smiles = smileCascade.detectMultiScale(faceROIGray, 4, 25)
                    for (sx, sy,sw,sh) in smiles:
                        if sy > height / 2:
                            cv2.rectangle(faceROIColored, (sx, sy), (sx+sw, sy+sh), (255, 255, 255), 2)
                            smiles_list.append({"y_min":int(sy),"y_max":int(sy+sh),"x_min":int(sx),"x_max":int(sx+sw)})
                        else:
                            pass
                    
                    frame_details["faces"] = faces_list
                    frame_details["eyes"] = eyes_list
                    frame_details["smiles"] = smiles_list
                    frame_details["gaze"] = gaze_label
                   
                detection_output.append(frame_details)
                
                fin_frame=True
                if fin_frame==True:
                  break
      except Exception as e:
        print(e)
        pass        


      
    blink_details = {'Blink count':total_blinks,'Duration':duration}  
    
    
    
    
    with open(metricas_video, "w") as outfile:
        json.dump(detection_output, outfile, ensure_ascii= False)
    print("Metrics file created")

    with open(metricas_parpadeos, "w") as outfile:
        json.dump(blink_details, outfile, ensure_ascii= False)
    print("Blink file created")

    with open(metricas_emociones, "w") as outfile:
        json.dump(emociones, outfile, ensure_ascii= False)
    print("Emotions file created")    

  else:
    print("Metrics file was already created")
  fin_ev_video = time.time()
  duracion_ev_video = round(fin_ev_video - inicio_ev_video, 2)
  return {'duracion':duracion_ev_video}