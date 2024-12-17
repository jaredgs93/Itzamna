import os
import moviepy.editor as mp

#Runtime Evaluation Libraries
import time
import datetime
import shutil
#Copy myprosody modified version
import myprosody2 as mysp
import pickle

#Folder MyProsody DON'T REMOVE
base_path = os.path.dirname(os.path.abspath(__file__))
folder_mysp = os.path.join(base_path, "Mysp", "myprosody-master", "myprosody")


def extraccion_audio(person_folder, video_url, person_id):
  inicio_extr_audio = time.time()
  try:
    video = mp.VideoFileClip(video_url)
    audio_video = os.path.join(person_folder, person_id+"_audio.wav")
    isExist = os.path.exists(audio_video)
    if not isExist:
      video.audio.write_audiofile(audio_video, fps=44100, nbytes=4, logger='bar')
      print("Audio obtained")
      #If the audio is larger than 25 MB, we reduce the size by fps to 22050.
      if os.path.getsize(audio_video) > 25000000:
        video.audio.write_audiofile(audio_video, fps=22050, nbytes=4, logger='bar')

        
    else:
      print("Audio was obtained previously")
  except Exception as e:
    print("Exception in audio extraction", e)
  fin_extr_audio = time.time()
  duracion_extr_audio = round(fin_extr_audio-inicio_extr_audio, 2)
  print('Audio extraction time:',duracion_extr_audio, 'secs.')
  #For prosody
  folder_mysp = os.path.join(base_path, 'Mysp', 'myprosody-master', 'myprosody')
  src_path = audio_video
  dst_path = os.path.join(folder_mysp,'dataset', 'audioFiles', person_id+'_audio.wav')
  shutil.copy(src_path, dst_path)

def general_statistics(person_id, person_folder):
    folder = os.path.abspath(folder_mysp)
    try:
        metrics = mysp.mysptotal(person_id+'_audio', folder)
        
        
        metrics.drop('number_ of_syllables', inplace=True, axis=1)
        metrics.drop('speaking_duration', inplace=True, axis=1)
        metrics.drop('original_duration', inplace=True, axis=1)
        
        metrics= metrics.to_dict()
        
        for i in metrics:
          try:
            metrics[i]= float(metrics[i][0])
          except:
            metrics[i]= metrics[i][0]

        #Mood
        mood_value = mysp.myspgend(person_id+'_audio',folder)
        metrics["mood"]=mood_value
    except Exception as e:
        print('Exception in prosody metrics extraction', e)
        pass
    #Moving the TextGrid file to the subject folder
    #If the person's id has a . we replace it with a _ as this is a change made by MyProsody.
    if '.' not in person_id and ' ' not in person_id:
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia)
    elif ' ' in person_id:
      person_id_provisional = person_id.replace(' ','_')
      #Rename the TextGrid file
      os.rename(os.path.join(folder, 'dataset', 'audioFiles', person_id_provisional+'_audio.TextGrid'), os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid'))
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia)
    else:
      person_id_provisional = person_id.replace('.','_')
      #Rename the TextGrid file
      os.rename(os.path.join(folder, 'dataset', 'audioFiles', person_id_provisional+'_audio.TextGrid'), os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid'))
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia) 
    return metrics