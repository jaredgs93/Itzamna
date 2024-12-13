import asyncio
import uvicorn
import gc
import psutil

collected = gc.collect()

#Import libraries
#For video format conversion if needed
import subprocess

from audio import *
from text import *
from video import *
from processing import *
from inference import *




#For the generation of unique IDs
import uuid


#For API creation
from fastapi import FastAPI,HTTPException


import cv2
from pydantic import BaseModel


import os
import ffmpeg
import json
import shutil


#FOLDERS THAT MAKE UP THE API
#For temp files
base_path = os.path.dirname(os.path.abspath(__file__))
temp_folder = os.path.join(base_path, 'temp')
#Dictionaries
dictionaries_folder = os.path.join(base_path, 'dictionaries')
#Mysp
Mysp_folder = os.path.join(base_path, 'Mysp', 'myprosody-master', 'myprosody', 'dataset', 'audioFiles')

class Video(BaseModel):
    video_url: str
    topic:str

class Rule(BaseModel):
    antecedent: str
    consequent: str
    consequent_value:str

import logging

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)

app = FastAPI()

metadata = {}

metricas_memoria = psutil.virtual_memory()

collected = gc.collect()
print("Garbage collector: collected","%d objects." % collected)



from fastapi import HTTPException
from fastapi import FastAPI
import ffmpeg

app = FastAPI()


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(f'{process_time:0.4f} sec')
    return response

@app.get('/')
def read_root():
    return ({"Welcome":'API for soft skills assessment in videos'})

#Endpoint to obtain possible antecedents for rules
@app.get('/antecedents')
def get_antecedents():
    try:
        variables_borrosas_collection = evaluacion_soft_skills_db['fuzzy_variables']
        antecedentes = list(variables_borrosas_collection.find({'$or':[{'metrica':True}, {'soft_skill':{'$exists':False}}]},{'_id':0}))
        #Changing the presentation format of variables
        antecedentes_en =[]
        for antecedente in antecedentes:
            print(antecedente)
            antecedente_en = {}
            antecedente_en['variable_description'] = antecedente['descripcion_variable_en']
            antecedente_en['variable_description_es'] = antecedente['descripcion_variable']
            antecedente_en['fuzzy_sets'] = antecedente['conjuntos_borrosos_en']
            antecedente_en['fuzzy_sets_es'] = antecedente['conjuntos_borrosos']
            antecedente_en['variable_name_es'] = antecedente['nombre_variable']
            if 'metrica' in antecedente:
                antecedente_en['measure'] = antecedente['metrica']
            antecedentes_en.append(antecedente_en)
        return antecedentes_en
    except Exception as e:
        return {"Error":str(e)}

#Endpoint to obtain soft skills
@app.get('/consequents')
def get_consequents():
    try:
        variables_borrosas_collection = evaluacion_soft_skills_db['fuzzy_variables']
        consecuentes = list(variables_borrosas_collection.find({'metrica':{'$exists':False}},{'_id':0}))
        #Changing the presentation format of variables
        consecuentes_en =[]
        for consecuente in consecuentes:
            print(consecuente)
            consecuente_en = {}
            consecuente_en['variable_description'] = consecuente['descripcion_variable_en']
            consecuente_en['variable_description_es'] = consecuente['descripcion_variable']
            consecuente_en['fuzzy_sets'] = consecuente['conjuntos_borrosos_en']
            consecuente_en['fuzzy_sets_es'] = consecuente['conjuntos_borrosos']
            consecuente_en['variable_name_es'] = consecuente['nombre_variable']
            consecuentes_en.append(consecuente_en)
        return consecuentes_en
    except Exception as e:
        return {"Error":str(e)}
    
#Endpoint for rule creation
@app.post('/create_rule', status_code=201)
def rule_post(rule:Rule):
    try:
        reglas_collection = evaluacion_soft_skills_db['rules_evaluation']
        rule_document = {'antecedente':rule.antecedent, 'consecuente':rule.consequent, 'consecuente_valor':rule.consequent_value}
        reglas_collection.insert_one(rule_document)
        #Regresamos un mensaje de éxito con estado 201
        return {"Message":"Rule created"}
    except Exception as e:
        return {"Error":str(e)}

#Endpoint to check if a video is eligible for evaluation
@app.post('/check_video_eligibility')
async def check_video_eligibility(video: Video):
    try:
        print("Attempting to retrieve video metadata.")
        video_metadata = ffmpeg.probe(video.video_url)

        # Look for the video stream in metadata
        stream_video = next((stream for stream in video_metadata['streams'] if stream['codec_type'] == 'video'), None)
        if not stream_video:
            print("No video stream found in metadata.")
            return {"eligible": False, "detail": "No valid video stream found."}

        # Extract video dimensions and duration
        height = int(stream_video['height'])
        width = int(stream_video['width'])
        area_video = height * width
        duration = float(video_metadata['format']['duration'])

        print(f"Video details - Duration: {duration}s, Resolution: {width}x{height}, Area: {area_video}")

        # Validate video based on duration and dimensions
        if duration >= 30 and duration <= 180 and area_video >= 921600:
            print("The video is valid for evaluation.")
            return {"eligible": True, "detail": "The video is valid for evaluation."}
        elif duration < 30:
            print("The video is too short, so it cannot be evaluated. The duration must be between 30 and 210 seconds.")
            return {"eligible": False, "detail": "The video is too short, so it cannot be evaluated. The duration must be between 30 and 210 seconds."}
        elif duration > 180:
            print("The video is too large, so it cannot be evaluated. The duration must be between 30 and 210 seconds.")
            return {"eligible": False, "detail": "The video is too large, so it cannot be evaluated. The duration must be between 30 and 210 seconds."}
        elif area_video < 921600:
            print(f"The video dimensions are very small ({width}x{height}), so it cannot be evaluated. The minimum dimensions of the video are 1280 x 720.")
            return {"eligible": False, "detail": f"The video dimensions are very small ({width}x{height}), so it cannot be evaluated. The minimum dimensions of the video are 1280 x 720."}

    except ffmpeg.Error as e:
        # Handle ffmpeg-specific errors (e.g., invalid URL, unsupported format)
        print('ffmpeg error:', e.stderr)
        return {"eligible": False, "detail": "Error processing video. Please check the URL and format."}

    except Exception as e:
        # Log any other unexpected errors with details for debugging
        print("Unexpected Exception:", str(e))
        return {"eligible": False, "detail": f"An unexpected error occurred: {str(e)}"}



from multiprocessing import Pool

#Endpoint to evaluate the soft skills of a video
@app.post('/evaluate_skills')
async def video_evaluation(video: Video):
    collected = gc.collect()
    print("Garbage collector: collected", "%d objects." % collected)

    try:
        # Check eligibility and capture the response
        eligibility_response = await check_video_eligibility(video)

        # If the video is not eligible, return a 422 error with the detail message
        if not eligibility_response["eligible"]:
            raise HTTPException(status_code=422, detail=eligibility_response["detail"])

        # If eligible, proceed with the evaluation
        resultado_skills = proceso_evaluacion_soft_skills(video)
        await asyncio.sleep(5)
        return resultado_skills

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions to preserve the status and message
        raise http_exc

    except Exception as e:
        # Handle unexpected errors with a generic message
        print("Unexpected Exception:", str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred during evaluation.")

def proceso_evaluacion_soft_skills(video:Video):
        try:
            global temp_folder
            temp_folder=os.path.join(temp_folder, 'Soft skills')
            video_url = video.video_url
            tema = video.topic
            print(video_url, tema)
            #Validate that within the temporary folder there is a subfolder called Soft skills
            if os.path.exists(temp_folder)==False:
                os.mkdir(temp_folder)
                print('Folder:', os.path.abspath(temp_folder))

            #Generate a unique ID for the person
            person_id = str(uuid.uuid1())
            print('Person ID',person_id)
            #Create the folder for the person
            person_folder = os.path.join(temp_folder, person_id)
            
            if os.path.exists(person_folder)==False:
                os.mkdir(person_folder)
                print('Folder:', os.path.abspath(person_folder))
                
            video_metadata = ffmpeg.probe(video_url)
            #Save the metadata of the video
            with open(os.path.join(person_folder, person_id+'_metadata.json'), 'w') as f:
                json.dump(video_metadata, f)
            duration = float(video_metadata['format']['duration'])
            print('Video duration: ',duration)

            #Verify if the video is in webm format
            if video_metadata['format']['format_name']=='matroska,webm':
                #Convert the video to mp4
                print('Converting to MP4')
                subprocess.run('ffmpeg -i """'+video_url+'""" '+'"""'+person_folder+person_id+'.mp4"""',shell=True)
                video_url = os.path.join(person_folder, person_id+'.mp4')
                #New metadata for the video
                video_metadata = ffmpeg.probe(video_url)


            #Extracting the number of frames included in the video
            
            try:
                nb_frames = int(video_metadata['streams'][0]['nb_frames'])
            except:
                nb_frames = False
            
            #If video has more than one frame, evaluate
            if nb_frames>5 or nb_frames==False:
                extraccion_audio(person_folder, video_url, person_id)
                extraccion_transcripcion(person_folder, person_id)
                metricas_texto(person_folder, person_id,duration, nb_frames)
                extraccion_temas(person_id,person_folder, tema)
                evaluacion_video(person_folder, person_id, video_url, duration)
                procesamiento_metricas(person_folder, person_id, duration, tema)
                resultado_skills = calculo_metricas(person_folder, person_id, tema)
            elif nb_frames>=1 and nb_frames<=5:
                extraccion_audio(person_folder, video_url, person_id)
                extraccion_transcripcion(person_folder, person_id)
                metricas_texto(person_folder, person_id,duration, nb_frames)
                extraccion_temas(person_id,person_folder, tema)
                procesamiento_metricas(person_folder, person_id, duration, tema)
                resultado_skills = calculo_metricas(person_folder, person_id, tema)
            #Remove the person folder
            try:
                shutil.rmtree(person_folder)
            except:
                pass
            
            #Remove the files created by Mysp
            try:    
                os.remove(os.path.join(Mysp_folder, person_id+'_audio.wav'))
            except:
                pass
            try:
                os.remove(os.path.join(Mysp_folder, person_id+'_audio.TextGrid'))
            except:
                pass

            print('End of process')
            #Temp folder is returned to its original value
            temp_folder = 'temp'
            
            collected = gc.collect()
            print("Garbage collector: collected","%d objects." % collected)
            
            return resultado_skills


        except Exception as e:
            #Return the temp folder to its original value
            temp_folder = 'temp'
            print('Excepción:',e)
            pass