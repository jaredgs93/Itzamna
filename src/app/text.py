from audio import general_statistics
import os
#Bibliotecas para la evaluación del tiempo de ejecución
import time
import datetime

import whisper

#MODELOS
#Modelo de Whisper para transcripción
whisper_model = whisper.load_model("base")
#Modelo de Whisper en inglés (se usa únicamente cuando el idioma según la transcripción no es inglés ni español)
whisper_model_en = whisper.load_model("base.en")
#Modelo de Whisper 'large'. Se usa para traducir los textos que no están ni en español ni en inglés al inglés
#whisper_model_large = whisper.load_model("large")

from decouple import config
from openai import OpenAI
client = OpenAI(api_key = config('OPENAI_API_KEY'))

import json
import textgrid
import pandas as pd
import spacy
from collections import Counter

from urllib import request, parse
import urllib.request, json
meaning_cloud_classification_url = 'https://api.meaningcloud.com/class-2.0'
meaning_cloud_sentiments_url = 'https://api.meaningcloud.com/sentiment-2.1'
from decouple import config
meaning_cloud_api_key = config('MEANINGCLOUD_API_KEY')

import pyphen
a = pyphen.Pyphen(lang='es')

#CARGA DE DICCIONARIOS DE DATOS (ESPAÑOL E INGLÉS)
#Se cargan los diccionarios para las transcripciones en español e inglés de forma global
#De esta manera se evita llenarlos por cada evaluación
#Dictionaries
dictionaries_folder = 'dictionaries'
folder_espanol = os.path.join(dictionaries_folder, 'es')
folder_ingles = os.path.join(dictionaries_folder, 'en')

#Se almacenarán los diccionarios según el idioma
idiomas=['es','en']
diccionarios ={'es':[],
               'en':[]}

#Se crea un ciclo donde se recuperan los diccionarios
for lang in idiomas:
  folder_path = os.path.join(dictionaries_folder, lang)

  #Conectores
  connectors_json = os.path.join(folder_path, 'connectors.json')
  with open(connectors_json, "r") as rf:
      connectors_list = json.load(rf)
  
  #Vicios
  vices_json = os.path.join(folder_path, 'language_vices.json')
  with open(vices_json, "r") as rf:
      vices_list = json.load(rf)

  #Muletillas
  fillers_json = os.path.join(folder_path, 'fillers.json')
  with open(fillers_json, "r") as rf:
      fillers_list = json.load(rf)
  
  #Concreción
  concretion_json = os.path.join(folder_path, 'concretion_terms.json')
  with open(concretion_json, "r") as rf:
      concretion_list = json.load(rf)
    
  #Respeto
  respect_json = os.path.join(folder_path, 'respect_terms.json')
  with open(respect_json, "r") as rf:
      respect_list = json.load(rf)
  
  #Orden
  order_json = os.path.join(folder_path, 'order_terms.json')
  with open(order_json, "r") as rf:
      order_list = json.load(rf)
    
  #Para tribunales amigables
  #Discurso
  speech_json = os.path.join(folder_path, 'speech.json')
  with open(speech_json, "r") as rf:
      discurso_list = json.load(rf)

  #Common words
  df_cw = pd.read_csv(os.path.join(folder_path, '1000_elements.txt'), sep ="\t")
  common_words = list(df_cw.Elemento.unique())

  
  diccionarios[lang] = [{'connectors':connectors_list,'language_vices':vices_list,'fillers':fillers_list,'concretion':concretion_list,'respect':respect_list,
                         'order':order_list,'common_words':common_words, 'speech':discurso_list}]
  #También se agrega el pipeline de SpaCy según el idioma del video a evaluar
  #Por otra parte, se agregan los modelos de clasificación de texto para meaning cloud
  #Los temas para la evaluación de topics se agregan
  if lang=='es':
    diccionarios[lang].append({'pipeline':'es_core_news_lg'})
    diccionarios[lang].append({'models_text_classification':['EUROVOC_es_ca']})
  if lang=='en':
    diccionarios[lang].append({'pipeline':'en_core_web_lg'})
    diccionarios[lang].append({'models_text_classification':['IPTC_en']})
  try:
    topics_to_detect = pd.read_csv(os.path.join(folder_path, 'topics.csv'))
    diccionarios[lang].append(topics_to_detect)
  except:
    pass

def traduccion_transcripcion(audio_path):
  audio_path= open(audio_path, "rb")
  translation = client.audio.translations.create(
    model="whisper-1", 
    response_format="verbose_json",
    file=audio_path
  )
  
  translation_dict = translation.to_dict()
  return translation_dict

def extraccion_transcripcion(person_folder, person_id):
  print('***Transcription extraction***')
  inicio_extr_transcr = time.time()
  audio_path = os.path.join(person_folder, person_id+'_audio.wav')
  #Inicia el procedimiento de transcripción
  transcription_video = os.path.join(person_folder, person_id+'_transcription.json')
  if os.path.exists(transcription_video)==False:
    transcription = whisper_model.transcribe(audio_path)
    
    if transcription['language']!='en' and transcription['language']!='es' and transcription['text']!='':
      #Versión 1: Extraer la transcripción usando el modelo en inglés
      #transcription = whisper_model_en.transcribe(audio_path)
      #Versión 2: Extraer la transcripción usando el modelo 'large'
      #transcription = whisper_model_large.transcribe(audio_path, task = 'translate')

      #Versión 3: Extraer la transcripción usando la API de OpenAI
      transcription = traduccion_transcripcion(audio_path)

      #Cambiar el idioma a inglés para evitar conflictos en métricas basadas en el idioma
      transcription['original_language'] = transcription['language']
      transcription['language'] = 'en'
    with open(transcription_video, "w") as outfile:
      json.dump(transcription, outfile, ensure_ascii= False)
      print("Transcription obtained")
  else:
    print('Trascription was obtained previously')
  fin_extr_transcr = time.time()
  duracion_extr_transcr = round(fin_extr_transcr-inicio_extr_transcr, 2)
  print('Transcription extraction time:',duracion_extr_transcr, 'secs.')

def metricas_texto(person_folder, person_id,duration, nb_frames):
  print('***Text metrics***')
  inicio_proc_texto = time.time()
  fichero_metricas = os.path.join(person_folder, person_id+'_text_measures.json')
  
  if os.path.exists(fichero_metricas)==False:
    #Métricas de prosodia
      
      transcripcion_json = os.path.join(person_folder, person_id+"_transcription.json")
      error_segmento=False
      #Se revisan los errores de los segmentos de la transcripción para su posterior corrección
      with open(transcripcion_json, "r") as rf:
        transcription_data = json.load(rf)
      #Revisión de los segmentos en reversa
      for segmento in reversed(transcription_data['segments']):
        #Verificar si el rango total del segmento está fuera de la duración
        if segmento['start']>duration:
          transcription_data['segments'].remove(segmento)
          transcription_data['text'] = transcription_data['text'].replace(segmento['text'], '')
          error_segmento = True
        #Verificar si el final del segmento es incorrecto
        elif segmento['start']<duration and segmento['end']>duration:
          segmento['end'] = duration
          error_segmento = True
      
      if error_segmento==True:
        with open(transcripcion_json, "w") as outfile:
          json.dump(transcription_data, outfile, ensure_ascii= False)
      print("Transcription was corrected")
      transcription_language = transcription_data['language']

      metricas_palabras = elementos_transcripcion(person_folder, person_id, duration, transcription_language)
      
      num_palabras = metricas_palabras["num_palabras"]
      metrics = {}
      
      if num_palabras>0 and (nb_frames>5 or nb_frames==False):
        try:
          inicio_prosodia = time.time()
          metrics = general_statistics(person_id, person_folder)
          fin_prosodia = time.time()
          #duracion_prosodia = fin_prosodia - inicio_prosodia
          #print('Duración de la prosodia:', formato_tiempo(duracion_prosodia))
          prosody_file = os.path.join(person_folder, person_id + '_prosody.TextGrid')
    
          tg = textgrid.TextGrid.fromFile(prosody_file)
          
          for intervalo in tg[1]:
            if intervalo.mark == 'sounding':
              inicio_discurso = round(intervalo.minTime,2)
              break

          metrics['inicio_discurso'] = inicio_discurso
          metrics['num_palabras'] = metricas_palabras['num_palabras']
          #Articulación se recalcula
          metrics['articulacion'] = silabas(transcripcion_json)
        except Exception as e:
           print(e, "Exception in prosody metrics extraction")
        
        
      """else:
        metrics = {'num_pausas': 0, 'articulacion': 0, 'rate_of_speech': 0, 
                    'balance': 0, 'f0_mean': 0, 'f0_std': 0,
                    'f0_median': 0, 'f0_min': 0, 'f0_max': 0, 
                    'f0_quantile25': 0, 'f0_quan75': 0, 'mood': 'Silent'}"""
      if metrics!= {}:
        metricas_palabras['prosodia']=metrics
      #Omitir resultados que no se escribirán en el fichero
      try:
        metricas_palabras.pop('inicio_discurso')
        metricas_palabras.pop('num_palabras')
        metricas_palabras.pop('duracion_minutos')
      except Exception as e:
         print(e, "Exception when removing prosody elements")
      
      with open(fichero_metricas, "w") as outfile:
        json.dump(metricas_palabras, outfile, ensure_ascii= False)
        print("Metrics file created")
      
  else:
    print("Metrics file was obtained previously")    
  fin_proc_texto = time.time()
  #duracion_proc_texto = fin_proc_texto - inicio_proc_texto - duracion_prosodia
  #return {'duracion_procesamiento_texto':duracion_proc_texto, 'duracion_prosodia':duracion_prosodia}

def elementos_transcripcion(person_folder, person_id, duration, transcription_language):

    #A partir del idioma de la transcripción se evaluarán las métricas
    lista_diccionarios = diccionarios[transcription_language][0]
    connectors_dictionary = lista_diccionarios['connectors']
    vices_dictionary = lista_diccionarios['language_vices']
    filles_dictionary = lista_diccionarios['fillers']
    concretion_dictionary = lista_diccionarios['concretion']
    respect_dictionary = lista_diccionarios['respect']
    order_dictionary = lista_diccionarios['order']
    common_words = lista_diccionarios['common_words']
    speech_dictionary = lista_diccionarios['speech']

    #Pipeline según el idioma
    nlp = spacy.load(diccionarios[transcription_language][1]['pipeline'])

    transcripcion_json = os.path.join(person_folder, person_id+'_transcription.json')

    adjectives=[]
    connectors = []
    vices = []
    fillers_l = []
    concretion_l = []
    respect_terms_l = []
    order_terms_l = []
    #Tribunales amigables
    speech_elements = []
    num_adjetivos_con_rep=0
    num_conectores_con_rep=0
    num_palabras_vagas_con_rep=0
    num_muletillas_con_rep = 0 
    num_term_concrecion_con_rep = 0
    duracion_minutos=0
    start_speech = -1
    with open (transcripcion_json) as f:
        data = json.load(f)
    speech_transcription = data['text'].lower()
        
    no_words = len(speech_transcription.split())

    
    if no_words==0:
        start_speech=0
        adjectives = []
        num_adjetivos_con_rep = 0
        connectors = []
        num_conectores_con_rep = 0
        vices = []
        num_palabras_vagas_con_rep = 0
        fillers_l = []
        num_muletillas_con_rep = 0
        duracion_minutos = 0
        concretion_l = []
        num_term_concrecion_con_rep = 0
        adjectives_l = []
        num_adjetivos_con_rep = 0
        respect_terms_l = []
        order_terms_l = []
        #Tribunales amigables
        speech_elements = {}
        
    else:
        tiempo_c_sonidos=0
        #Validar tiempo en silencio
        
        for segmento in data['segments']:
          tiempo_c_sonidos = tiempo_c_sonidos+ (segmento['end']-segmento['start'])
        

        if tiempo_c_sonidos/duration>0.1:
            
            #Validamos si las palabras existen

            doc = nlp(speech_transcription)
            words = [token.text
                  for token in doc
                    if not token.is_stop and not token.is_punct and ["\n", ""]]
            word_freq = Counter(words)
            freqs = [word for word in word_freq.keys() if word in common_words ]
            if len(freqs)==0:
                no_words=0
                start_speech=0
                adjectives = []
                num_adjetivos_con_rep = 0
                connectors = []
                num_conectores_con_rep = 0
                vices = []
                num_palabras_vagas_con_rep = 0
                fillers_l = []
                num_muletillas_con_rep = 0
                duracion_minutos = 0
                concretion_l = []
                num_term_concrecion_con_rep = 0
                adjectives_l = []
                num_adjetivos_con_rep = 0
                respect_terms_l = []
                order_terms_l = []
                #Tribunales amigables
                speech_elements = {}
            
            else:
            
              #Se validará el texto
              duracion_minutos=round(data['segments'][-1]['end'],2)
              #duracion_minutos=round(duracion_minutos/60000, 2)
              start_speech = data['segments'][0]['start']
              #start_speech=start_speech/1000
              
              #Detección de adjetivos
              document = nlp(speech_transcription)
              
              for token in document:
                if token.pos_ =='ADJ':
                  try:
                      existe = next(item for item in adjectives if item["adjetivo"]==token.lemma_)
                      existe["conteo"]+=1
                  except:
                      adjectives.append({"adjetivo":token.lemma_, "conteo":1})
              #Conteo de adjetivos
              for a in adjectives:
                  num_adjetivos_con_rep = num_adjetivos_con_rep+a["conteo"]

              #Detección de conectores
              for item in connectors_dictionary:
                  if item in speech_transcription:
                    connectors.append({"conector":item, "conteo":speech_transcription.count(item)})
              
              #Conteo de conectores
              for c in connectors:
                  num_conectores_con_rep=num_conectores_con_rep+c["conteo"]
              
              #Detección de palabras vagas
              for t in vices_dictionary:
                  for v in vices_dictionary[t]:
                      if v in speech_transcription.split():
                          vices.append({"término":v, "conteo":speech_transcription.split().count(v)})
      
              #Conteo de palabras vagas
              for v in vices:
                  num_palabras_vagas_con_rep = num_palabras_vagas_con_rep+v["conteo"]
             
              
              #Detección de mueletillas
              for f in filles_dictionary:
                  if f in speech_transcription:
                      fillers_l.append({"muletilla":f, "conteo":speech_transcription.count(f)})

              #Conteo de muletillas
              for m in fillers_l:
                  num_muletillas_con_rep=num_muletillas_con_rep+m["conteo"]
              
              
              #Detección de términos de concreción
              for ct in concretion_dictionary:
                for t in ct['terminos']:
                  if speech_transcription.count(t) > 0 :
                    concretion_l.append({'termino':t, 'conteo':speech_transcription.count(t)})
              
              #Conteo de términos de concreción
              for ct in concretion_l:
                num_term_concrecion_con_rep = num_term_concrecion_con_rep+ct['conteo']

              #Detección de términos de respeto
              respect_terms_l = {'saludo':[s for s in respect_dictionary['saludo'] if s in speech_transcription], 'despedida':[d for d in respect_dictionary['despedida'] if d in speech_transcription]}
              
              #Detección de términos de orden
              order_terms_l = {'inicio':[o for o in order_dictionary['inicio'] if o in speech_transcription],
              'intermedio':[o for o in order_dictionary['intermedio'] if o in speech_transcription],
              'final':[o for o in order_dictionary['final'] if o in speech_transcription]}

              #Tribunales amigables
              try:
                speech_l = {'verguenza':[v for v in speech_dictionary['verguenza'] if v in speech_transcription],
                            'negacion':[v for v in speech_dictionary['negacion'] if v in speech_transcription],
                            'sorpresa':[v for v in speech_dictionary['sorpresa'] if v in speech_transcription],
                            'miedo':[v for v in speech_dictionary['miedo'] if v in speech_transcription],
                            'enojo':[v for v in speech_dictionary['enojo'] if v in speech_transcription]}
              except:
                 pass
       
            
      
    metricas_texto =  {"inicio_discurso":start_speech, 
            "num_palabras": no_words, 
            "adjetivos":adjectives, 
            "conectores":connectors, 
            "palabras_vagas":vices, 
            "muletillas":fillers_l, 
            "concrecion":concretion_l ,
            "terminos_respeto": respect_terms_l,
            "terminos_orden": order_terms_l,
            #"elementos_discurso": speech_l,
            "duracion_minutos":duracion_minutos}
    return metricas_texto

def silabas(transcription):
  try:
    with open(transcription, "r") as rf:
      decoded_data = json.load(rf)
    transcription = decoded_data['text']
    segundo_inicial = decoded_data['segments'][0]['start']
    segundo_final = decoded_data['segments'][-1]['end']
    duracion = segundo_final-segundo_inicial
    duracion = round(duracion, 2)

    silabas = a.inserted(transcription)
    silabas = silabas.split('-')
    silabas = [x for x in silabas if x!=' ']
    num_silabas = len(silabas)
    
    silabas_p_segundo = num_silabas/duracion
    
  except Exception as e:
    silabas_p_segundo = 0
    
    print('Exception in the calculation of syllables',e)
  return round(silabas_p_segundo,2)

def extraccion_temas(person_id,person_folder, tema):
  inicio_extr_temas = time.time()
  #Detectar si el fichero de temas ya existe
  temas_json = os.path.join(person_folder, person_id + '_topics.json')
  if os.path.exists(temas_json)==False:
    #Crear el diccionario para almacenar los temas
    temas_dict = {'EUROVOC':{'category_list':[]}}
    #Leer el fichero de transcripción
    json_transcripcion = os.path.join(person_folder, person_id+'_transcription.json')
    with open(json_transcripcion, "r") as rf:
      decoded_data = json.load(rf)
    
    #Usamos el texto para detectar los temas
    transcripcion = decoded_data ['text']

    #Extraer los temas usando la API de ChatGPT
    temas = extraccion_temas_gpt(transcripcion, tema)
    #Convertir el texto a un diccionario
    temas = json.loads(temas)
    temas_dict['EUROVOC']['category_list'] = temas

    """
    #Usamos el idioma para indicar el modelo de meaningcloud a utilizar
    idioma_transcripcion = decoded_data['language']
    modelos = diccionarios[idioma_transcripcion][2]['models_text_classification']
    #Se obtienen los temas según los modelos
    for modelo in modelos:
      header = parse.urlencode({'key':meaning_cloud_api_key,
                              'model': modelo,
                              'txt':transcripcion}).encode()
      req =  request.Request(meaning_cloud_classification_url, data=header)
      with urllib.request.urlopen(req) as url:
          temas = json.loads(url.read().decode())
          temas.pop('status')
      temas_dict[modelo] = temas
    """
    #Se escribe el fichero de temas
    with open(temas_json, "w") as outfile:
        json.dump(temas_dict, outfile, ensure_ascii= False)
        print("Topics file created")
  else:
    print('Topics file was obtained previously')
  fin_extr_temas = time.time()
  duracion_extr_temas = fin_extr_temas - inicio_extr_temas
  return {'duracion':duracion_extr_temas}


def extraccion_temas_gpt(transcripcion, tema):
      completion = client.chat.completions.create(
      messages = [{'role': 'system', 'content' : """You are a text classifier. Your task consist of classifying the following text into the EuroVoc vocabulary. """+
                                                 """In addition, you will be provided with a topic which is being evaluated from the content of the text and you have to indicate the level of relevance of each detected topic to the topic to be evaluated."""+
                                                  """The output must be a JSON file with this structure. [{topic1}, {topic2},...,{topicn}]. Don't start with ```json\n. I need just the list. 
                                                    The output should be in English, with this structure: code: Which is the EuroVoc code, label: the label of the category, relevance: A value between 0 and 100, relevance_to_the_topic: A value between 0 and 100"""+
                                                  """The list must be ordered in descending order of relevance."""},
                  {'role': 'user', 'content' : f'text: {transcripcion}, topic: {tema}'}],
  
      model = 'gpt-4o'
  )
    
        
      chat_response = completion.choices[0].message.content
      return chat_response

def analisis_sentimientos(person_id,person_folder):
  inicio_analisis_sentimientos = time.time()
  #Detectar si el fichero de temas ya existe
  analisis_sentimientos_json = os.path.join(person_folder, person_id + '_analisis_sentimientos.json')
  if os.path.exists(analisis_sentimientos_json)==False:
    #Leer el fichero de transcripción
    json_transcripcion = os.path.join(person_folder, person_id+'_transcription.json')
    with open(json_transcripcion, "r") as rf:
      decoded_data = json.load(rf)
    
    #Usamos el texto para detectar los temas
    transcripcion = decoded_data ['text']
    
    #Consulta a la API de MeaningCloud
    header = parse.urlencode({'key':meaning_cloud_api_key,
                              'model': 'general',
                              'txt':transcripcion}).encode()
    req =  request.Request(meaning_cloud_sentiments_url, data=header)
    with urllib.request.urlopen(req) as url:
        resultado_analisis_sentimientos = json.loads(url.read().decode())
        resultado_analisis_sentimientos.pop('status')
    
    
    #Se escribe el fichero de temas
    with open(analisis_sentimientos_json, "w") as outfile:
        json.dump(resultado_analisis_sentimientos, outfile, ensure_ascii= False)
        print("El fichero de análisis de sentimientos se ha creado")
  else:
    print('Fichero de temas existente')
  fin_analisis_sentimientos = time.time()
  duracion_analisis_sentimientos = fin_analisis_sentimientos - inicio_analisis_sentimientos
  return {'duracion':duracion_analisis_sentimientos}