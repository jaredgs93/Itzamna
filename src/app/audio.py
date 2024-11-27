import os
import moviepy.editor as mp

#Bibliotecas para la evaluación del tiempo de ejecución
import time
import datetime
import shutil
#Copiar myprosody versión modificada
import myprosody2 as mysp
import pickle

#Folder MyProsody NO REMOVER
folder_mysp = "Mysp/myprosody-master/myprosody"


def extraccion_audio(person_folder, video_url, person_id):
  inicio_extr_audio = time.time()
  try:
    video = mp.VideoFileClip(video_url)
    audio_video = os.path.join(person_folder, person_id+"_audio.wav")
    isExist = os.path.exists(audio_video)
    if not isExist:
      video.audio.write_audiofile(audio_video, fps=44100, nbytes=4, logger='bar')
      print("Audio obtained")
      #Si el audio pesa mas de 25 MB, reducimos el tamaño por medio de los fps a 22050
      if os.path.getsize(audio_video) > 25000000:
        video.audio.write_audiofile(audio_video, fps=22050, nbytes=4, logger='bar')

        
    else:
      print("Audio was obtained previously")
  except Exception as e:
    print("Exception in audio extraction", e)
  fin_extr_audio = time.time()
  duracion_extr_audio = round(fin_extr_audio-inicio_extr_audio, 2)
  print('Audio extraction time:',duracion_extr_audio, 'secs.')
  #Para prosodia
  folder_mysp = os.path.join('Mysp', 'myprosody-master', 'myprosody')
  src_path = audio_video
  dst_path = os.path.join(folder_mysp,'dataset', 'audioFiles', person_id+'_audio.wav')
  shutil.copy(src_path, dst_path)

def general_statistics(person_id, person_folder):
    folder = os.path.abspath(folder_mysp)
    try:
        metrics = mysp.mysptotal(person_id+'_audio', folder)
        
        
        metrics.drop('number_ of_syllables', inplace=True, axis=1)
        #metrics.drop('rate_of_speech', inplace=True, axis=1)
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
    #Mover el ficheto TextGrid a la carpeta del sujeto
    #Si el id de la persona tiene un . lo reemplazamos por un _ ya que es un cambio que hace MyProsody
    if '.' not in person_id and ' ' not in person_id:
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia)
    elif ' ' in person_id:
      person_id_provisional = person_id.replace(' ','_')
      #Cambiamos el nombre del archivo TextGrid
      os.rename(os.path.join(folder, 'dataset', 'audioFiles', person_id_provisional+'_audio.TextGrid'), os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid'))
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia)
    else:
      person_id_provisional = person_id.replace('.','_')
      #Cambiamos el nombre del archivo TextGrid
      os.rename(os.path.join(folder, 'dataset', 'audioFiles', person_id_provisional+'_audio.TextGrid'), os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid'))
      textgrid_original = os.path.join(folder, 'dataset', 'audioFiles', person_id+'_audio.TextGrid')
      textgrid_copia = os.path.join(person_folder, person_id+'_prosody.TextGrid')
      shutil.copy(textgrid_original, textgrid_copia) 
    return metrics