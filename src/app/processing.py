import os
import json

import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

import textgrid

#Runtime Evaluation Libraries
import time
import datetime

from text import diccionarios
import spacy
from collections import Counter

#For the assessment of congruence
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances

# tokenize and pad every document to make them of the same size
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

#To translate texts from Spanish into English
from translation import translator

# loading pre-trained embeddings, each word is represented as a 300 dimensional vector
base_path = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(base_path, 'models')
import gensim
W2V_PATH = os.path.join(models_folder, 'GoogleNews-vectors-negative300.bin.gz')
model_w2v = gensim.models.KeyedVectors.load_word2vec_format(W2V_PATH, binary=True)

#GLOVE
embeddings_index = dict()
with open(os.path.join(models_folder, 'glove.6B.300d.txt'), encoding="utf8") as file:
  for line in file:
      values = line.split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

# Function for measuring the average similarity between transcript fragments
def most_similar(doc_id,similarity_matrix,matrix, documents_df):
    mean_similarity = 0
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    
    for ix in similar_ix:
        mean_similarity+=similarity_matrix[doc_id][ix]
    mean_similarity = mean_similarity/(len(similarity_matrix)-1)
   
    return mean_similarity


def horizontal(width, x):
  posicion_horizontal = ctrl.Antecedent(np.arange(0,width , 0.01), 'posicion_horizontal')
  posicion_horizontal['Izquierda'] = fuzz.trapmf(posicion_horizontal.universe, [0, 0, (width/2)/2, width/2])
  posicion_horizontal['Centro'] = fuzz.trimf(posicion_horizontal.universe, [(width/2)/2, width/2, (width/2)+(width/2)/2])
  posicion_horizontal['Derecha'] = fuzz.trapmf(posicion_horizontal.universe, [width/2, (width/2)+(width/2)/2,float("inf") ,float("inf")])
 

  grados_pertenencia = {}
  for t in posicion_horizontal.terms: 
    mval = np.interp(x, posicion_horizontal.universe, posicion_horizontal[t].mf)
    grados_pertenencia[t] =  mval
 
  new = max(grados_pertenencia, key=grados_pertenencia.get)
  
  return new

def vertical(height, x):
  posicion_vertical = ctrl.Antecedent(np.arange(0,height , 0.01), 'posicion_vertical')
  posicion_vertical['Arriba'] = fuzz.trapmf(posicion_vertical.universe, [0, 0, (height/2)/2, height/2])
  posicion_vertical['Centro'] = fuzz.trimf(posicion_vertical.universe, [(height/2)/2, height/2, (height/2)+(height/2)/2])
  posicion_vertical['Abajo'] = fuzz.trapmf(posicion_vertical.universe, [height/2, (height/2)+(height/2)/2,float("inf") ,float("inf")])
  

  grados_pertenencia = {}
  for t in posicion_vertical.terms: 
    mval = np.interp(x, posicion_vertical.universe, posicion_vertical[t].mf)
    grados_pertenencia[t] =  mval
  
  new = max(grados_pertenencia, key=grados_pertenencia.get)
 
  return new

def escala_emocion (x):
  escala_emocion = ctrl.Antecedent(np.arange(-10, 10, 0.01), 'escala_emocion')
  escala_emocion['Alto'] = fuzz.trimf(escala_emocion.universe, [-10, 10, 10])
 

  grados_pertenencia = {}
  
  mval = np.interp(x, escala_emocion.universe, escala_emocion['Alto'].mf)
  return (mval)*10

def tono_grave (x):
  tono_grave = ctrl.Antecedent(np.arange(97, 163, 0.01), 'tono_grave')
  
  tono_grave['Alto'] = fuzz.trimf(tono_grave.universe, [97, 163, 163])
  mval = np.interp(x, tono_grave.universe, tono_grave['Alto'].mf)
  return (mval)*10

def tono_agudo(x):
  tono_agudo = ctrl.Antecedent(np.arange(163, 245, 0.01), 'tono_agudo')
  tono_agudo['Alto'] = fuzz.trimf(tono_agudo.universe, [163, 245, 245])
  mval = np.interp(x, tono_agudo.universe, tono_agudo['Alto'].mf)
  return (mval)*10 

def evaluacion_ruido(transcription_file, prosody_file):
  duracion_sonido = 0
  duracion_ruido = 0
  intervalos_audio = []
  mega_segmentos = []
  try:
    constancia_frecuencia = 0
    ruido = 0
    with open(transcription_file, "r") as rf:
      decoded_data = json.load(rf)
    transcription_text = decoded_data['text']
    segmentos = decoded_data['segments']
    #Obtaining the intervals 
    tg = textgrid.TextGrid.fromFile(prosody_file)
    num_intervals = len(tg[1].intervals)

    for inter in tg[1]:
      start = inter.minTime
      end = inter.maxTime
      duration = end-start
      status = inter.mark
      intervalos_audio.append({'start':start, 'end':end, 'duration':duration, 'status':status})
    
    for inter in intervalos_audio:
      #If the segment has sound and lasts more than one second or if it is a pause in the speech (less than 2 seconds).
      if (inter['status']=='sounding'and inter['duration']>=1) or (inter['status']=='silent' and inter['duration']<=2 and intervalos_audio.index(inter)<len(intervalos_audio)-1 and intervalos_audio.index(inter)!=0):
        constancia_frecuencia = constancia_frecuencia + inter['duration']
      #If the segment has sound and lasts less than one second it is considered noise
      elif inter['status']=='sounding'and inter['duration']<1:
        ruido+=1
       
    
    
    first_sounding = next(inter for inter in intervalos_audio if inter['status']=='sounding')
    first_sounding = first_sounding['start']
    
    last_sounding = next(inter for inter in reversed(intervalos_audio) if inter['status']=='sounding')
    last_sounding = last_sounding['end']
    
    #Evaluate consistency with the length of the speech, not the length of the video
    constancia_frecuencia = (constancia_frecuencia*10)/(last_sounding-first_sounding)
    
    
    ruido = ruido/len(intervalos_audio)
  except Exception as e:
    print('Exception in noise:',e)
    ruido = 10
    constancia_frecuencia = 0
  return {'constancia_expresion_ideas':constancia_frecuencia, 'nivel_ruido':ruido}

# Metrics processing for inference
moods = [{'original':'Showing no emotion, normal', 'etiqueta':'Normal'}, {'original':'Reading', 'etiqueta':'Reading'}, 
          {'original':'Speaking passionately', 'etiqueta':'Passionately'}]

def procesamiento_metricas(person_folder, person_id, duration, tema):
  print('***Measures processing***')
  inicio_procesamiento_metricas = time.time()
  metricas_para_inferencia = os.path.join(person_folder, person_id+'_measures_for_inference.json')
  if os.path.exists(metricas_para_inferencia)==False:
    metricas_obtenidas = metricas_sujeto(person_id, person_folder, duration, tema)
    with open(metricas_para_inferencia, "w") as outfile:
      json.dump(metricas_obtenidas, outfile, ensure_ascii= False)
    print("Measures obtained")
  else:
    print('Measures already obtained')
  fin_procesamiento_metricas = time.time()
  duracion_procesamiento_metricas = round(fin_procesamiento_metricas - inicio_procesamiento_metricas,2)
  print('Measures processing time:', duracion_procesamiento_metricas)

def metricas_sujeto(person_id, person_folder, duration, tema):
  # Pre-generated metrics files
  metadata_json =  os.path.join(person_folder, person_id+'_metadata.json')
  transcripcion_json = os.path.join(person_folder, person_id+'_transcription.json')
  metricas_discurso_json =  os.path.join(person_folder, person_id+'_text_measures.json')
  prosody_file = os.path.join(person_folder, person_id+'_prosody.TextGrid')
  temas_json = os.path.join(person_folder, person_id+'_topics.json')
  analisis_sentimientos_json = os.path.join(person_folder, person_id+'_analisis_sentimientos.json')
  rostros_json = os.path.join(person_folder, person_id+'_faces_smiles.json')
  parpadeos_json = os.path.join(person_folder, person_id+'_blinking.json')
  emociones_json = os.path.join(person_folder, person_id+'_emotions.json')
  idioma_transcripcion = ''
  duration_mins = round(duration/60,2)

  
  # Before processing the metrics, it is checked that the subject's face and speech have been detected in the video
  # Quality starts with the label 'Baja' (Low)
  calidad = 'Baja'
  frames_c_rostro = 0
  #Detect that the speech has words
  try:
    with open(transcripcion_json, "r") as rf:
      decoded_data = json.load(rf)
    texto_transcripcion = decoded_data['text']
    palabras_transcripcion = len(texto_transcripcion.split())
  except Exception as e:
    print('Exception in words detection:',e)
  
  #Detect that the face has been detected in at least one frame
  try:
    with open(rostros_json, "r") as rf:
      decoded_data = json.load(rf)
   
    for frame in decoded_data:
      if 'faces' in frame and len(frame['faces'])>0:
        frames_c_rostro +=1
  except Exception as e:
    print('Exception in face detection:',e)
  # If the video has a speech with words and the face has been detected in at least one frame, the video is of High quality
  if palabras_transcripcion>3 and frames_c_rostro>0:
    calidad = 'Alta'
  # If no face was detected in any frame, the video is of Medium quality
  
  #elif frames_c_rostro==0 and palabras_transcripcion>3:
  elif frames_c_rostro>0 and palabras_transcripcion>3:
    calidad = 'Media'
  elif palabras_transcripcion>3 and os.path.exists(rostros_json)==False:
    calidad = 'Fake'
  
  # The metrics are calculated from the files
 
  if calidad!='':
   
    resultados = {'sujeto':person_id, 'tema':tema, 'calidad':calidad, 'duracion':duration,
                  'ruido':0.14, 'fluidez':20, 'mood':'Normal','uniformidad_voz':5, 'rapidez_del_habla':5, 'constancia':9, 'tiempo_de_reaccion':1.5, 
                  'adecuacion_al_tema':5, 'congruencia':5, 'concrecion':5, 'densidad':5, 'ejemplos':15, 'metrica_ejemplos':5, 'muletillas':2.25, 'orden':5, 'expresion':5,
                  'organizacion':4, 'originalidad':5, 'no_redundancia':5, 'respeto':5, 'tiempo_verbal':5, 'vaguedad': 3, 'cantidad':2, 'relevancia_temas':50, 'temas_diferentes':0,
                  'peso_mood':6, 'vocabulario':5, 'ideas':5, 'argumentacion_liderazgo':5,
                  'cambios_de_postura':5, 'gestos_anomalos':5, 'mirada':5, 'parpadeo':5, 'sonrisas':4, 'serenidad':5, 'emocion_negativa':5, 'emocion_positiva':5, 'emocion_estres':5,
                  'mirada_furtiva':5, 'lenguaje_no_verbal':5}
    
    # Speech metrics/prosody processing
    try:
      # Read json with speech metrics
      with open(metricas_discurso_json, "r") as rf:
          decoded_data = json.load(rf)
      # Adding metrics that do not require processing to the results
      try:
        tiempo_de_reaccion = decoded_data['prosodia']['inicio_discurso']
        resultados['tiempo_de_reaccion'] = tiempo_de_reaccion
      except Exception as e:
        print('Exception in reaction time detection:',e)
        pass
      try:
        resultados['fluidez'] = int(decoded_data['prosodia']['num_pausas'])
      except Exception as e:
        print('Exception in fluency evaluation:',e) 
        pass
      try:
        # Speech speed is measured in syllables per second
        resultados['rapidez_del_habla'] = decoded_data['prosodia']['articulacion']
      except Exception as e:
        print('Excepción en la evaluación de rapidez del habla:',e)
        pass
      try:
        # Normalise the value for the number of connectors per minute
        no_conectores = len(decoded_data['conectores'])
        no_conectores_p_min = no_conectores/duration_mins
        resultados['organizacion'] = round(no_conectores_p_min, 2)
      except Exception as e:
        print('Exception in organization evaluation:',e)
        pass

      # Assign the corresponding mood label so that in the inference phase it is adapted to the defined labels
      try:
        mood = decoded_data['prosodia']['mood']
        mood = next(item for item in moods if item['original']==mood)
        mood = mood['etiqueta']
        resultados['mood'] = mood
      except Exception as e:
        print('Exception in mood evaluation:',e)
        pass

      # Normalise the value of the different vague words per minute
      try:
        no_palabras_vagas = len(decoded_data['palabras_vagas'])
        no_palabras_vagas_p_min = no_palabras_vagas/duration_mins
        resultados['vaguedad'] = round(no_palabras_vagas_p_min, 2)
      except Exception as e:
        print('Exception in vagueness evaluation:',e)
        pass

      # Normalise the value of distinct adjectives per minute
      # Examples are calculated as the number of adjectives per minute
      try:
        no_adjetivos = len(decoded_data['adjetivos'])
        ejemplos = no_adjetivos/duration_mins
        ejemplos = round(ejemplos, 2)
        resultados['ejemplos'] = ejemplos
        # Example metrics are calculated, i.e. scaled from 0 to 10 for the aggregations where it is required
        
        metrica_ejemplos = min(10, ejemplos*10/20)
        resultados['metrica_ejemplos'] = metrica_ejemplos
      except Exception as e:
        print('Exception in examples evaluation:',e)
        pass

      # Normalise the value of the number of different crutches per minute.
      try:
        no_muletillas = len(decoded_data['muletillas'])
        no_muletillas_p_min = round(no_muletillas/duration_mins,2)
        resultados['muletillas'] = no_muletillas_p_min
      except Exception as e:
        print('Exception in fillers evaluation:',e)
        pass
      # Terms of concretion per minute
      try:
        no_concrecion = 0
        for termino in decoded_data['concrecion']:
          no_concrecion+=termino['conteo']
        no_concrecion_p_min = round(no_concrecion/duration_mins,2)
      except Exception as e:
        print('Exception in the concreteness evaluation:',e)
        pass

      # Respect metric. Only greetings and farewells are considered
      try:
        respeto = 0
        # If the person salutes, 5 points are added
        if len(decoded_data['terminos_respeto']['saludo'])>0:
          respeto+=5
        # If the person says goodbye, 5 points are added
        if len(decoded_data['terminos_respeto']['despedida'])>0:
          respeto+=5
        resultados['respeto'] = respeto
      except Exception as e:
        print('Exception in respect evaluation:',e)
        pass

      #Order metric
      try:
        orden = 0
        # If a start order connector is used, add 3 points
        if len(decoded_data['terminos_orden']['inicio'])>0:
          orden+=3
        # If using an end order connector, add 3 points
        if len(decoded_data['terminos_orden']['final'])>0:
          orden+=3
        # If a connector of order of medium is used, add 3 points
        if len(decoded_data['terminos_orden']['intermedio'])>0:
          orden+=2
        orden = min(orden, 10)
        resultados['orden'] = orden
      except Exception as e:
        print('Exception in order evaluation:',e)
        pass

      #Voice tone
      # It identifies whether the tone the subject uses is low or high pitched
      try:
        tono_de_voz = 0
        tipo_voz=''
        # The tone of voice depends on the average f0 of the speech
        f0_mean = decoded_data['prosodia']['f0_mean']
        
        # A set of conditions is defined where depending on the mood and the mean fundamental frequency the type of voice is established
        if mood=='Normal':
          if f0_mean>97 and f0_mean<=114:
            tipo_voz = 'Grave'
          elif f0_mean>163 and f0_mean<=197:
            tipo_voz = 'Aguda'
        elif mood=='Reading':
          if f0_mean>114 and f0_mean<=135:
            tipo_voz = 'Grave'
          elif f0_mean>197 and f0_mean<=226:
            tipo_voz = 'Aguda'
        elif mood == 'Passionately':
          if f0_mean>135 and f0_mean<=163:
            tipo_voz = 'Grave'
          elif f0_mean>226 and f0_mean<=245:
            tipo_voz = 'Aguda'
        
        # Once the voice type is obtained, the voice tone is calculated.
        if tipo_voz=='Grave':
          tono_de_voz = tono_grave(f0_mean)
        elif tipo_voz == 'Aguda':
          tono_de_voz = tono_agudo(f0_mean)
        resultados['uniformidad_voz'] = tono_de_voz
      except Exception as e:
        print('Exception in tone/voice uniformity evaluation: ',e)
        pass
     
    except Exception as e:
      print('Exception in metrics processing, speech metrics',e)
    
    # Processing metrics based on discourse content
    try:
      with open(transcripcion_json, "r") as rf:
        decoded_data = json.load(rf)
      texto_transcripcion = decoded_data['text']
      # Transcription language for NLP
      idioma_transcripcion = decoded_data['language']
      nlp = spacy.load(diccionarios[idioma_transcripcion][1]['pipeline'])

      # Calculation of the originality
      try:
        doc = nlp(texto_transcripcion)
        # Getting the words of the speech
        words = [token.text
            for token in doc
                if not token.is_stop and not token.is_punct and ["\n", ""]]
        # Counting the frequency of each word in the speech
        word_freq = Counter(words)
        # Load the most frequent words according to the language of transcription
        common_words = diccionarios[idioma_transcripcion][0]['common_words']
        # Identify the words in the discourse that belong to the most frequent words in their language
        freqs = [word for word in word_freq.keys() if word in common_words ]
        originalidad = np.round((10* (len(word_freq.keys()) - len(freqs))) / len(word_freq.keys()),0)
        resultados['originalidad'] = originalidad
      except Exception as e:
        print('Exception in originality evaluation:',e)
        pass

      #Redundance
      # Five most frequently used tokens
      try:
        word_freq = Counter(words)
        repet = word_freq.most_common(1)[0][1]
        no_redundancia = 10-repet
       
        resultados['no_redundancia'] = no_redundancia
      except Exception as e:
        print('Exception in redundancy evaluation:',e)
        pass

      #Verbs
      try:
        lst_verbs = []
        
        for token in doc:
          if token.pos_ == "VERB":
            
            tense = token.morph.get("Tense")
            lst_verbs.append(tense if tense != [] else ["Inf"])
        verbs_freq = Counter([item for sublist in lst_verbs for item in sublist])
        tiempo_verbal = np.round((verbs_freq['Pres'] * 10) / len(lst_verbs),0)
        resultados['tiempo_verbal'] = tiempo_verbal
      except Exception as e:
        print('Exception in verb tense evaluation',e)
        pass
     

      #Congruence
      try:
        # First count how many dots are used to separate ideas
        contador_puntos = texto_transcripcion.count('.')
       
        # If there are ellipsis points (they represent a crutch, as has been detected in some videos, discount the point counter)
        contador_p_suspensivos = texto_transcripcion.count('...')

        # Update the points counter
        contador_puntos = contador_puntos - (contador_p_suspensivos*3)

        # If the text of the transcript has no full stops or is the end, the text segments will be used
        if contador_puntos==0 or (contador_puntos==1 and texto_transcripcion[-1:]=='.'):
            # If in English, use segments as obtained with Whisper
            if idioma_transcripcion=='en':
                
                segments = [seg['text'] for seg in decoded_data['segments']]
            # In other languages, translate the segments
            else:
                segments = [translator(seg['text']) for seg in decoded_data['segments']]
                print('segments', segments)
        # If the text has several points, i.e. several ideas, divide by point
        else:
            # If in English, split the transcript
            if idioma_transcripcion=='en':
                
                segments = texto_transcripcion.split('. ')
            # In other languages, translate into English and split the transcript
            else:
                segments = translator(texto_transcripcion).split('. ')
                print(segments)

  
        # Procedure based on WORD2VEC
        
        # For WordEmbedding, it is necessary to segment the transcript into sentences
        # Split the text using the full stop
        
        segments_df=pd.DataFrame(segments,columns=['segments'])
        
        
        # Remove special characters and stop words
        stop_words_l=stopwords.words('english')
        segments_df['segments_cleaned']=segments_df.segments.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
       

        tfidfvectoriser=TfidfVectorizer()
        tfidfvectoriser.fit(segments_df.segments_cleaned)
        tfidf_vectors=tfidfvectoriser.transform(segments_df.segments_cleaned)

        tokenizer=Tokenizer()
        tokenizer.fit_on_texts(segments_df.segments_cleaned)
        tokenized_documents=tokenizer.texts_to_sequences(segments_df.segments_cleaned)
        tokenized_paded_documents=pad_sequences(tokenized_documents, padding='post')
        
        vocab_size=len(tokenizer.word_index)+1
        
        # Creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
        embedding_matrix=np.zeros((vocab_size,300))
        for word,i in tokenizer.word_index.items():
            if word in model_w2v:
                embedding_matrix[i]=model_w2v[word]
        # Creating document-word embeddings
        document_word_embeddings=np.zeros((len(tokenized_paded_documents),len(tokenized_paded_documents[0]),300))
        for i in range(len(tokenized_paded_documents)):
            for j in range(len(tokenized_paded_documents[0])):
                document_word_embeddings[i][j]=embedding_matrix[tokenized_paded_documents[i][j]]
        document_word_embeddings.shape

        # Calculating average of word vectors of a document weighted by tf-idf
        document_embeddings=np.zeros((len(tokenized_paded_documents),300))
        words=tfidfvectoriser.get_feature_names()
        
        for i in range(len(document_word_embeddings)):
            for j in range(len(words)):
                try:
                  document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors.toarray()[i][j]
                except:
                  pass
        metrica_cohesion_word2vec = 0
        pairwise_similarities=cosine_similarity(document_embeddings)
        
        for index, row in segments_df.iterrows():  
          media_similitudes_word2vec = most_similar(index,pairwise_similarities,'Cosine Similarity', segments_df)
          metrica_cohesion_word2vec+=media_similitudes_word2vec
        metrica_cohesion_word2vec = round((metrica_cohesion_word2vec*10)/len(pairwise_similarities),2)
        
        
        #GLOVE
        # Creating embedding matrix, every row is a vector representation from the vocabulary indexed by the tokenizer index. 
        embedding_matrix=np.zeros((vocab_size,300))

        for word,i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                
        # Calculating average of word vectors of a document weighted by tf-idf
        document_embeddings=np.zeros((len(tokenized_paded_documents),300))
        words=tfidfvectoriser.get_feature_names()

        # Instead of creating document-word embeddings, directly creating document embeddings
        for i in range(segments_df.shape[0]):
            for j in range(len(words)):
                document_embeddings[i]+=embedding_matrix[tokenizer.word_index[words[j]]]*tfidf_vectors.toarray()[i][j]
        metrica_cohesion_glove = 0
        pairwise_similarities=cosine_similarity(document_embeddings)
        
        for index, row in segments_df.iterrows():  
            media_similitudes_glove = most_similar(index,pairwise_similarities,'Cosine Similarity', segments_df)
            metrica_cohesion_glove+=media_similitudes_glove
        
        metrica_cohesion_glove = round((metrica_cohesion_glove*10)/len(pairwise_similarities),2)
        
        congruencia = (metrica_cohesion_word2vec + metrica_cohesion_glove)/2
        resultados['congruencia'] = congruencia
      except Exception as e:
        print('Exception in congruence evaluation: ',e)
        pass

    except Exception as e:
      print('Exception in metrics processing, speech content: ',e)
      pass
    
    # Processing from the textgrid
    if os.path.exists(prosody_file):
      metricas_ruido = evaluacion_ruido(transcripcion_json, prosody_file)
      resultados['constancia'] = metricas_ruido['constancia_expresion_ideas']
      resultados['ruido'] = metricas_ruido['nivel_ruido']

    # Processing from themes
   
    try:
      with open(temas_json, "r") as rf:
        decoded_data = json.load(rf)
      if tema!='':
        topic = 0
        n_topics = 0
        # The list of categories depends on the language and models used
        try:
          category_list = decoded_data["EUROVOC"]["category_list"]
        except:
          if idioma_transcripcion == 'es':
            category_list = decoded_data["EUROVOC_es_ca"]["category_list"]
            
          if idioma_transcripcion == 'en':
            category_list = decoded_data['IPTC_en']["category_list"]

        num_temas = len(category_list)
        cantidad = 0
        if num_temas>0:
          cantidad = round((np.log2(num_temas)*2)+1,0) 
        
        resultados['cantidad'] = cantidad
       
        # If GPT was used to identify the themes, it is assessed against the themes obtained

        for item in category_list:
          n_topics += 1
          if int(item['relevance_to_the_topic'])>=70:
            topic = int(item["relevance"]) if int(item["relevance"]) > topic else topic
            n_topics -=1

        # If MeaningCloud was used, it is evaluated with the dictionary with the available topics
        if 'EUROVOC_es_ca' in decoded_data or 'IPTC_en' in decoded_data:
          # The list of topics to be detected depends on the language used
          topics_to_detect = diccionarios[idioma_transcripcion][3]
          

          specific_topics_to_detect = topics_to_detect[topics_to_detect['tema'] == tema]
        
          for item in category_list:
            n_topics += 1
            if int(item["code"]) in specific_topics_to_detect.values:
              topic = int(item["relevance"]) if int(item["relevance"]) > topic else topic
              n_topics -=1 # subtract because it is relevant
        resultados['relevancia_temas'] = topic
        resultados['adecuacion_al_tema'] = round(topic/10,0)
        resultados['temas_diferentes'] = n_topics
        
        #Density
        if num_temas>0:
          resultados['densidad'] = min((np.log2(num_temas)*4)+1,10)
        else:
          resultados['densidad'] = 0
      
    except Exception as e:
      print('Exception in topic metrics processing: ',e)
      pass

    
    try:
      if resultados['temas_diferentes']>0:
        #Organization -> Connectors per minute
        concrecion = min(no_concrecion_p_min + (10 * math.sqrt(1/resultados['temas_diferentes'])),10)
        resultados['concrecion'] = concrecion
    except Exception as e:
      print('Exception in the calculation of concreteness: ',e)
      pass
  

    #Mood
    try:
      peso_mood = 0
      if resultados['mood'] == 'Reading':
        peso_mood = 3
      if resultados['mood'] == 'Normal':
        peso_mood = 6
      if resultados['mood'] == 'Passionately':
        peso_mood = 10
      resultados['peso_mood'] = peso_mood
    except Exception as e:
      print('Exception in mood weight calculation: ',e)
      pass


    #Expression
    try:
      expresion = (0.4*resultados['concrecion']) + (0.2*resultados['metrica_ejemplos']) + (0.1*resultados['peso_mood'])+ (0.3*resultados['adecuacion_al_tema']) 
      resultados['expresion'] = expresion
    except Exception as e:
      print('Exception in the calculation of the expression: ',e)  
      pass
    try:
      # If the mood weight is 10, of passionate, then the expression is 0
      if resultados['peso_mood']!=10:
        expresion = 0
      else:
        expresion = (0.6*resultados['concrecion']) + (0.4*resultados['peso_mood'])
      resultados['expresion_estabilidad_emocional'] = expresion
    except Exception as e:
      print('Excepción en el cálculo de la expresión para neuroticismo: ',e)  
      pass


    # Text quality
    # Requires adjectives and crutches
    metrica_adjetivos = 0
    if resultados['ejemplos']!=0:
      metrica_adjetivos = round(np.log2(resultados['ejemplos'])*2,0)
    
    metrica_muletillas = 0
    if resultados['muletillas']>=4:
      metrica_muletillas = 0
    else:
      if resultados['muletillas']==0:
        metrica_muletillas=10
      else:
        metrica_muletillas = 2**resultados['muletillas']
    
    try:
      vocabulario = round(0.25*resultados['no_redundancia']+0.125* resultados['originalidad'] +0.25*metrica_adjetivos+0.25*metrica_muletillas+0.125*resultados['tiempo_verbal'],2)
      resultados['vocabulario'] = vocabulario
    except Exception as e:
      print('Exception in the calculation of the vocabulary: ',e)
      pass

    try:
      ideas = 0
      # Ideas are measured by quantity, appropriateness to the subject and reaction time
      if resultados['cantidad']>=3:
        ideas+=2.5
      if resultados['cantidad']>=2 and resultados['cantidad']<3:
        ideas+=1.25
      if resultados['cantidad']>0 and resultados['cantidad']<2:
        ideas+=0.5
      if resultados['cantidad']==0:
        ideas+=0
      
      ideas+=resultados['adecuacion_al_tema']*0.5

      # Evaluation with reaction time
      if resultados['tiempo_de_reaccion']<=1:
        ideas+=2.5
      if resultados['tiempo_de_reaccion']>1 and resultados['tiempo_de_reaccion']<=1.5:
        ideas+=1.25
      if resultados['tiempo_de_reaccion']>1.5 and resultados['tiempo_de_reaccion']<=2.5:
        ideas+=0.5
      if resultados['tiempo_de_reaccion']>2.5:
        ideas+=0
      resultados['ideas'] = ideas
    except Exception as e:
      print('Exception in the calculation of ideas: ',e)

    # Metrics from video files are calculated only if a face has been detected
    
    if frames_c_rostro>0:
      frames_c_sonrisa = 0
      metrica_sonrisa = 0
      mirada_fija = 0
      # The dimensions of the video are extracted from the video metadata
      # Read the metadata file
      with open(metadata_json, "r") as rf: 
        decoded_data = json.load(rf)
      # Get the dimensions of the video
      video_stream = next(item for item in decoded_data["streams"] if item["codec_type"] == "video")
      width =video_stream['width']
      height = video_stream['height']

      try:
        with open(rostros_json, "r") as rf:
          decoded_data = json.load(rf)
        frames_analizados = len(decoded_data)
        # A list is created to store the detected areas of the faces in the frames
        areas_rostro = []
        # Three lists are created to record movements
        movimientos_horizontales = []
        movimientos_verticales = []
        movimientos_foco = []
        for frame in decoded_data:
          # Frames with smiles are detected
          if 'smiles' in frame and len(frame['smiles'])>0:
            
            frames_c_sonrisa+=1
          if 'faces' in frame and len(frame['faces'])>0:
            # Calculation of the area of the faces for position
            x_min = frame['faces'][0]['x_min']
            x_max = frame['faces'][0]['x_max']
            y_min = frame['faces'][0]['y_min']
            y_max = frame['faces'][0]['y_max']
            longitud_x = x_max - x_min
            longitud_y = y_max - y_min
            area_rostro = longitud_x * longitud_y
            areas_rostro.append(area_rostro)

            # For the horizontal position of the face
            if width>height:
              x_min_h = horizontal(width, x_min)
              x_max_h = horizontal(width, x_max)
            else:
              x_min_h = horizontal(height, x_min)
              x_max_h = horizontal(height, x_max)
            posicion_h = ''
            if x_min_h==x_max_h:
              posicion_h = x_min_h
            elif x_min_h== 'Izquierda' and x_max_h=='Derecha':
              posicion_h = x_min_h+'/Centro/'+x_max_h
            else:
              posicion_h = x_min_h+'/'+x_max_h
            if len(movimientos_horizontales)==0:
              movimientos_horizontales.append(posicion_h)
              
            else:
              if movimientos_horizontales[len(movimientos_horizontales)-1]!=posicion_h:
                movimientos_horizontales.append(posicion_h)
            
            # For the vertical position of the face
            if height>width:
              y_min_v = vertical(height, y_min)
              y_max_v = vertical(height, y_max)
            else:
              y_min_v = vertical(width, y_min)
              y_max_v = vertical(width, y_max)
            posicion_v = ''
            if y_min_v==y_max_v:
              posicion_v = y_min_v
            elif y_min_v== 'Arriba' and y_max_v=='Abajo':
              posicion_v = y_min_v+'/Centro/'+y_max_v
            else:
              posicion_v = y_min_v+'/'+y_max_v
            if len(movimientos_verticales)==0:
              movimientos_verticales.append(posicion_v)
            else:
              if movimientos_verticales[len(movimientos_verticales)-1]!=posicion_v:
                movimientos_verticales.append(posicion_v)
          # For the centered gaze indicator, detect whether the gaze in the frame is centred
          if 'gaze' in frame and frame['gaze']=='Center':
            mirada_fija+=1
          
        # Normalise the smile metric, scale 0 to 10.
        metrica_sonrisa = round((frames_c_sonrisa*10)/frames_c_rostro, 0)
        resultados['sonrisas'] = metrica_sonrisa
        #Normalizamos la métrica de mirada, escala 0 a 10
        metrica_mirada = (mirada_fija*10)/frames_analizados
        resultados['mirada'] = metrica_mirada
        mirada_furtiva = 10-metrica_mirada
        resultados['mirada_furtiva'] = mirada_furtiva

        media_area_rostro = sum(areas_rostro)/len(areas_rostro)
        media_area_rostro = round(media_area_rostro, 2)
        min_area_rostro = min(areas_rostro)
        max_area_rostro = max(areas_rostro)
       

        # Assess the areas of the face in each frame to determine whether it is far away, centred or near
        for area in areas_rostro:
          posicion_f=''
          # Assessment between the minimum and the average
          if area>=min_area_rostro and area<=media_area_rostro:
            distancia_al_min = area-min_area_rostro
            distancia_a_media = media_area_rostro-area
            if distancia_al_min<distancia_a_media:
              posicion_f = 'Lejos'
            else:
              posicion_f = 'Centro'
          
          # Evaluation between average and maximum
          if area>media_area_rostro and area<=max_area_rostro:
            distancia_a_media = area_rostro-media_area_rostro
            distancia_al_max = max_area_rostro-area
            if distancia_a_media<distancia_al_max:
              posicion_f = 'Centro'
            else:
              posicion_f = 'Cerca'
          if len(movimientos_foco)==0:
            movimientos_foco.append(posicion_f)
          else:
            if movimientos_foco[len(movimientos_foco)-1]!=posicion_f:
              movimientos_foco.append(posicion_f)
       

        # Metrics of movements per minute
        # Movements are calculated on a scale of 0-10
        # It is divided by 180, which is the maximum number of movements per minute from all angles
        movimientos = ((len(movimientos_horizontales) + len(movimientos_verticales) + len(movimientos_foco))*10)/180
        resultados['cambios_de_postura'] = movimientos
      except Exception as e:
        print('Exception in processing metrics, faces and smiles: ',e)
        pass
      #DEL FICHERO DE PARPADEOS
      try:
        with open(parpadeos_json, "r") as pf:
          parpadeos_data = json.load(pf)
        num_parpadeos = parpadeos_data['Blink count']
        # Normalisation of blinks (flickers) per minute metric
        # As 1 frame per second is analysed, the maximum number of flickers per second is 60
        # The flicker metric is on a scale of 0-10, so the value of the metric should be normalised
        metrica_parpadeos =min(round((num_parpadeos/duration_mins),2),10)
        resultados['parpadeo'] = metrica_parpadeos
      except Exception as e:
        print('Exception in blinking processing:',e)
      
      # From emotions file
      # A dictionary is created to count the dominant emotions detected
      l_emociones = {'angry': 0, 'sad': 0, 'neutral': 0, 'disgust': 0, 'surprise': 0, 'fear': 0, 'happy': 0}
      try:
        with open(emociones_json, "r") as rf:
          detalle_emociones = json.load(rf)

        # List of dominant emotions
        emociones_dominantes = [item ['dominant_emotion'] for item in detalle_emociones]
        # Calculation of the emotional stability metric
        estabilidad_emocional = 0
        for i in range(len(emociones_dominantes)):
          # Validate if the current element is equal to the following element
          if i<len(emociones_dominantes)-1 and emociones_dominantes[i] == emociones_dominantes[i-1]:
              
              estabilidad_emocional += 1
        # Normalise the value of emotional stability to be on the 0-1 scale
        emociones_mapeo = {'angry':0, 'disgust':1,'sad':2, 'neutral':3, 'fear':4,'surprise':5,'happy':6}
        # Converting the list of emotions into numbers
        lista_emociones_numeros = [emociones_mapeo[emocion] for emocion in emociones_dominantes]
        resultados['estabilidad_emocional'] = np.std(lista_emociones_numeros)/np.mean(lista_emociones_numeros)
        
        
          
      except Exception as e:
        print('Exception in emotion processing:',e)
        pass
     
      try:
        # Normalise the emotions in the dictionary on a scale of 0 to 10
        for e in l_emociones:
          l_emociones[e] = round((l_emociones[e]*10)/len(detalle_emociones),2)
       
        # Calculating negative emotion
        emocion_negativa = l_emociones['angry'] + l_emociones['disgust'] + l_emociones['fear'] + l_emociones['sad']-l_emociones['happy']-l_emociones['neutral']
        emocion_negativa = escala_emocion (emocion_negativa)
        resultados['emocion_negativa'] = emocion_negativa
        # Calculating positive emotion
        emocion_positiva = l_emociones['happy'] + l_emociones['neutral'] - l_emociones['angry'] - l_emociones['disgust'] - l_emociones['fear'] - l_emociones['sad']
        emocion_positiva = escala_emocion (emocion_positiva)
        resultados['emocion_positiva'] = emocion_positiva
        resultados['serenidad'] = emocion_positiva
        # Calculating stressed emotion
        emocion_estres = l_emociones['angry'] + l_emociones['fear'] - l_emociones['happy']
        emocion_estres = escala_emocion(emocion_estres)
        resultados['emocion_estres'] = emocion_estres
        gestos_anomalos = min(10-(l_emociones['fear']+l_emociones['angry'])+l_emociones['happy'],10)
        resultados['gestos_anomalos'] = gestos_anomalos

       
        resultados['expresion_facial_sorpresa'] = l_emociones['surprise']
        resultados['expresion_facial_miedo'] = l_emociones['fear']
        resultados['expresion_facial_enojo'] = l_emociones['angry']


      except Exception as ee:
        print(ee)

    #Non-verbal language
    try:
      if calidad!='Fake':
        lenguaje_no_verbal = 0
        
        # First, it is validated whether the subject is reading. The mood weight for Reading is 3.
        if resultados['peso_mood']==3:
          lenguaje_no_verbal = 1.5
        # If the person is not reading, validate with blinks, diffuse gazes, facial movements and analysis of abnormal gestures.
        # This procedure is omitted if the subject's face was not detected
        elif frames_c_rostro>0:
          # Depending on the rank to which each metric belongs, it is assigned a weighting
          # Blinking
          if metrica_parpadeos>=0 and metrica_parpadeos<2.5:
            lenguaje_no_verbal+=1.5
          elif resultados['parpadeo']>=2.5 and resultados['parpadeo']<=6.5:
            lenguaje_no_verbal+=3
          else:
            lenguaje_no_verbal+=0
          # Diffuse gazes
          if resultados['mirada_furtiva']>=0 and resultados['mirada_furtiva']<2.5:
            lenguaje_no_verbal+=3
          elif resultados['mirada_furtiva']>=2.5 and resultados['mirada_furtiva']<=5:
            lenguaje_no_verbal+=2
          elif resultados['mirada_furtiva']>5 and resultados['mirada_furtiva']<9:
            #1 point
            lenguaje_no_verbal+=1
          else:
            lenguaje_no_verbal+=0
          #Movements
          if resultados['cambios_de_postura']>=0 and resultados['cambios_de_postura']<=2:
            lenguaje_no_verbal+=2
          elif resultados['cambios_de_postura']>2 and resultados['cambios_de_postura']<=6:
            lenguaje_no_verbal+=1
          else:
            lenguaje_no_verbal+=0
          # Abnormal gestures
          if resultados['gestos_anomalos']>=0 and resultados['gestos_anomalos']<3:
            lenguaje_no_verbal+=0
          elif resultados['gestos_anomalos']>=3 and resultados['gestos_anomalos']<=8:
            lenguaje_no_verbal+=1
          else:
            lenguaje_no_verbal+=2
          
        resultados['lenguaje_no_verbal'] = lenguaje_no_verbal
    except Exception as e:
      print('Exception in non-verbal language calculation: ',e)
      pass

    # Non-verbal languaje (Alternative)
    try:
      lenguaje_no_verbal_no_desinformador = 0
      
      if frames_c_rostro>0:
        if metrica_parpadeos>=0 and metrica_parpadeos<2.5:
          lenguaje_no_verbal_no_desinformador+=1.5
        elif resultados['parpadeo']>=2.5 and resultados['parpadeo']<=6.5:
          lenguaje_no_verbal_no_desinformador+=3
        else:
          lenguaje_no_verbal_no_desinformador+=0
        if resultados['mirada_furtiva']>=0 and resultados['mirada_furtiva']<2.5:
          lenguaje_no_verbal_no_desinformador+=3
        elif resultados['mirada_furtiva']>=2.5 and resultados['mirada_furtiva']<=5:
          lenguaje_no_verbal_no_desinformador+=2
        elif resultados['mirada_furtiva']>5 and resultados['mirada_furtiva']<9:
          #1 point
          lenguaje_no_verbal_no_desinformador+=1
        else:
          lenguaje_no_verbal_no_desinformador+=0
        #Movements
        if resultados['cambios_de_postura']>=0 and resultados['cambios_de_postura']<=2:
          lenguaje_no_verbal_no_desinformador+=2
        elif resultados['cambios_de_postura']>2 and resultados['cambios_de_postura']<=6:
          lenguaje_no_verbal_no_desinformador+=1
        else:
          lenguaje_no_verbal_no_desinformador+=0
        if resultados['gestos_anomalos']>=0 and resultados['gestos_anomalos']<3:
          lenguaje_no_verbal_no_desinformador+=0
        elif resultados['gestos_anomalos']>=3 and resultados['gestos_anomalos']<=8:
          lenguaje_no_verbal_no_desinformador+=1
        else:
          lenguaje_no_verbal_no_desinformador+=2
        
      resultados['lenguaje_no_verbal_no_desinformador'] = lenguaje_no_verbal_no_desinformador
    except Exception as e:
      print('Excepción en cálculo de lenguaje no verbal de no desinformador:',e)
      pass

    # Calculation of leadership argumentation
    
    try:
      argumentacion_liderazgo = resultados['adecuacion_al_tema']*0.25 + resultados['orden']*0.25 + resultados['concrecion']*0.2 + resultados['metrica_ejemplos']*0.2 + resultados['respeto']*0.1
      resultados['argumentacion_liderazgo'] = argumentacion_liderazgo
    except:
      print('Exception in leadership argumentation calculation')
      pass


  # Finally, the numerical results are rounded.
  for r in resultados:
    try:
      resultados[r] = round(resultados[r],2)
    except:
      pass
    

  return resultados
