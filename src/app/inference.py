import os
#Bibliotecas para el procedimiento borroso
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import pandas as pd

#Bibliotecas para la evaluación del tiempo de ejecución
import time
import datetime

from collections import Counter

import json

#Para la traducción de texto
from googletrans import Translator
translator = Translator()

#Para la generación del reporte
from report_generation import *

from decouple import config
from pymongo import MongoClient
#Conexión a la base de datos
#mongodb_user = config('MONGODB_USER')
#mongodb_password = config('MONGODB_PASSWORD')
#uri = f'mongodb+srv://{mongodb_user}:{mongodb_password}@cluster0.aqov9.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
#client = MongoClient(uri)
client = MongoClient()
#Base de datos
evaluacion_soft_skills_db = client['soft_skills_evaluation']

#Datasets
dictionaries_folder = 'dictionaries'

#Diccionario para las explicaciones
with open(os.path.join(dictionaries_folder, 'variables_description.json'), "r") as rf:
  descripcion_variables = json.load(rf)

#Definimos los conjuntos borrosos de las variables. Deben crearse desde la primera ejecución 

#Conjuntos borrosos para la preevaluación
duracion = ctrl.Antecedent(np.arange(0,300 , 0.01), 'duracion')
duracion['Corta'] = fuzz.trimf(duracion.universe, [0, 0, 90])
duracion['Adecuada'] = fuzz.trapmf(duracion.universe, [0, 90, 150, 240])
duracion['Larga'] = fuzz.trapmf(duracion.universe, [150, 240, float("inf") ,float("inf")])

dimension_video = ctrl.Antecedent(np.arange(0,930 , 0.01), 'dimension_video')
dimension_video['Limitada'] = fuzz.trapmf(dimension_video.universe, [0, 0, 76.8,172.8])
dimension_video['Adecuada'] = fuzz.trapmf(dimension_video.universe, [76.8,172.8, float("inf") ,float("inf")])

calidad_video = ctrl.Consequent(np.arange(0,10 , 0.01), 'calidad_video')
calidad_video['Baja'] = fuzz.trimf(calidad_video.universe, [0, 0, 10])
calidad_video['Alta'] = fuzz.trimf(calidad_video.universe, [0, 10, 10])

#Conjuntos borrosos para la evaluación
#Tiempo de reacción
tiempo_de_reaccion = ctrl.Antecedent(np.arange(0,4 , 0.01), 'tiempo_de_reaccion')
tiempo_de_reaccion['Alto'] = fuzz.trapmf(tiempo_de_reaccion.universe, [float("-inf") ,float("-inf"), 1, 1.5])
tiempo_de_reaccion['Medio'] = fuzz.trimf(tiempo_de_reaccion.universe, [1, 1.5, 2.5])
tiempo_de_reaccion['Bajo'] = fuzz.trapmf(tiempo_de_reaccion.universe, [1.5, 2.5,float("inf") ,float("inf")])

#Fluidez
fluidez = ctrl.Antecedent(np.arange(0,50 , 0.01), 'fluidez')
fluidez['Alta'] = fuzz.trapmf(fluidez.universe, [float("-inf") ,float("-inf"), 10, 20])
fluidez['Media'] = fuzz.trimf(fluidez.universe, [10, 20, 30])
fluidez['Baja'] = fuzz.trapmf(fluidez.universe, [20, 30, float("inf") ,float("inf")])

#Rapidez del habla
rapidez_del_habla = ctrl.Antecedent(np.arange(0,8 , 0.01), 'rapidez_del_habla')
rapidez_del_habla['Baja'] = fuzz.trapmf(rapidez_del_habla.universe, [float("-inf") ,float("-inf"), 4, 5])
rapidez_del_habla['Media'] = fuzz.trimf(rapidez_del_habla.universe, [4, 5, 6])
rapidez_del_habla['Alta'] = fuzz.trapmf(rapidez_del_habla.universe, [5, 6, float("inf") ,float("inf")])

#Organización
organizacion = ctrl.Antecedent(np.arange(0,20 , 0.01), 'organizacion')
organizacion['Baja'] = fuzz.trapmf(organizacion.universe, [float("-inf") ,float("-inf"), 2, 4])
organizacion['Media'] = fuzz.trimf(organizacion.universe, [2, 4, 8])
organizacion['Alta'] = fuzz.trapmf(organizacion.universe, [4, 8, float("inf") ,float("inf")])

#Tono de voz (mood)
mood = ctrl.Antecedent(np.arange(0,5 , 0.01), 'mood')
mood['Reading'] = fuzz.trimf(mood.universe, [0, 1, 2])
mood['Normal'] = fuzz.trimf(mood.universe, [1, 2, 3])
mood['Passionately'] = fuzz.trimf(mood.universe, [2, 3, 4])

vaguedad = ctrl.Antecedent(np.arange(0,10 , 0.01), 'vaguedad')
vaguedad['Baja'] = fuzz.trapmf(vaguedad.universe, [float("-inf") ,float("-inf"), 2, 3])
vaguedad['Media'] = fuzz.trimf(vaguedad.universe, [2, 3, 6])
vaguedad['Alta'] = fuzz.trapmf(vaguedad.universe, [3, 6, float("inf") ,float("inf")])

#Ejemplos
ejemplos = ctrl.Antecedent(np.arange(0,50 , 0.01), 'ejemplos')
ejemplos['Bajo'] = fuzz.trapmf(ejemplos.universe, [float("-inf") ,float("-inf"), 10, 15])
ejemplos['Medio'] = fuzz.trimf(ejemplos.universe, [10, 15, 20])
ejemplos['Alto'] = fuzz.trapmf(ejemplos.universe, [15, 20, float("inf") ,float("inf")])

#Muletillas
muletillas = ctrl.Antecedent(np.arange(0,10 , 0.01), 'muletillas')
muletillas['Baja'] = fuzz.trapmf(muletillas.universe, [float("-inf") ,float("-inf"), 1.5, 3])
muletillas['Alta'] = fuzz.trapmf(muletillas.universe, [1.5, 3, float("inf") ,float("inf")])

#velocidad
velocidad = ctrl.Consequent(np.arange(0,10 , 0.01), 'velocidad')
velocidad['Baja'] = fuzz.trapmf(velocidad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
velocidad['Media'] = fuzz.trimf(velocidad.universe, [2.5, 5, 7.5])
velocidad['Alta'] = fuzz.trapmf(velocidad.universe, [5, 7.5, float("inf") ,float("inf")])

#firmeza
firmeza = ctrl.Consequent(np.arange(0,10 , 0.01), 'firmeza')
firmeza['Baja'] = fuzz.trapmf(firmeza.universe, [float("-inf") ,float("-inf"), 2.5, 5])
firmeza['Media'] = fuzz.trimf(firmeza.universe, [2.5, 5, 7.5])
firmeza['Alta'] = fuzz.trapmf(firmeza.universe, [5, 7.5, float("inf") ,float("inf")])

#Claridad
claridad = ctrl.Consequent(np.arange(0,10 , 0.01), 'claridad')
claridad['Baja'] = fuzz.trapmf(claridad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
claridad['Media'] = fuzz.trimf(claridad.universe, [2.5, 5, 7.5])
claridad['Alta'] = fuzz.trapmf(claridad.universe, [5, 7.5, float("inf") ,float("inf")])

#Densidad
densidad = ctrl.Antecedent(np.arange(0,10 , 0.01), 'densidad')
densidad['Baja'] = fuzz.trapmf(densidad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
densidad['Media'] = fuzz.trimf(densidad.universe, [2.5, 5, 7.5])
densidad['Alta'] = fuzz.trapmf(densidad.universe, [5, 7.5, float("inf") ,float("inf")])

#Concisión
concision = ctrl.Consequent(np.arange(0,10 , 0.01), 'concision')
concision['Baja'] = fuzz.trapmf(concision.universe, [float("-inf") ,float("-inf"), 2.5, 5])
concision['Media'] = fuzz.trimf(concision.universe, [2.5, 5, 7.5])
concision['Alta'] = fuzz.trapmf(concision.universe, [5, 7.5, float("inf") ,float("inf")])

#Decision making
decision_making = ctrl.Consequent(np.arange(0,10 , 0.01), 'decision_making')
decision_making['Bajo'] = fuzz.trapmf(decision_making.universe, [float("-inf") ,float("-inf"), 2.5, 5])
decision_making['Medio'] = fuzz.trimf(decision_making.universe, [2.5, 5, 7.5])
decision_making['Alto'] = fuzz.trapmf(decision_making.universe, [5, 7.5, float("inf") ,float("inf")])

sonrisa = ctrl.Antecedent(np.arange(0, 10, 0.01), 'sonrisa')
sonrisa['Escasa'] = fuzz.trapmf(sonrisa.universe, [float("-inf") ,float("-inf"), 3, 4])
sonrisa['Normal'] = fuzz.trimf(sonrisa.universe, [3, 4, 7])
sonrisa['Frecuente'] = fuzz.trapmf(sonrisa.universe, [4, 7, 10 ,10])

mirada = ctrl.Antecedent(np.arange(0, 10, 0.01), 'mirada')
mirada['Difusa'] = fuzz.trapmf(mirada.universe, [float("-inf") ,float("-inf"), 3, 5])
mirada['Normal'] = fuzz.trimf(mirada.universe, [3, 5, 8])
mirada['Centrada'] = fuzz.trapmf(mirada.universe, [5, 8, 10 ,10])

cambios_de_postura = ctrl.Antecedent(np.arange(0, 10, 0.01), 'cambios_de_postura')
cambios_de_postura['Pocos'] = fuzz.trapmf(cambios_de_postura.universe, [float("-inf") ,float("-inf"), 2.5, 5])
cambios_de_postura['Regular'] = fuzz.trimf(cambios_de_postura.universe, [2.5, 5, 7.5])
cambios_de_postura['Muchos'] = fuzz.trapmf(cambios_de_postura.universe, [5, 7.5, float("inf") ,float("inf")])

#Empatía
empatia = ctrl.Consequent(np.arange(0,10 , 0.01), 'empatia')
empatia['Baja'] = fuzz.trapmf(empatia.universe, [float("-inf") ,float("-inf"), 2.5, 5])
empatia['Normal'] = fuzz.trimf(empatia.universe, [2.5, 5, 7.5])
empatia['Alta'] = fuzz.trapmf(empatia.universe, [5, 7.5, float("inf") ,float("inf")])

#Expresión
expresion = ctrl.Antecedent(np.arange(0,10, 0.01), 'expresion')
expresion['Mala'] = fuzz.trapmf(expresion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
expresion['Normal'] = fuzz.trimf(expresion.universe, [2.5, 5, 7.5])
expresion['Buena'] = fuzz.trapmf(expresion.universe, [5, 7.5, float("inf") ,float("inf")])

#Calidad de la argumentación
calidad_argumentacion = ctrl.Consequent(np.arange(0,10 , 0.01), 'calidad_argumentacion')
calidad_argumentacion['Baja'] = fuzz.trapmf(calidad_argumentacion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
calidad_argumentacion['Media'] = fuzz.trimf(calidad_argumentacion.universe, [2.5, 5, 7.5])
calidad_argumentacion['Alta'] = fuzz.trapmf(calidad_argumentacion.universe, [5, 7.5, float("inf") ,float("inf")])

#Argumentación
argumentacion = ctrl.Consequent(np.arange(0,10 , 0.01), 'argumentacion')
argumentacion['Mala'] = fuzz.trapmf(argumentacion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
argumentacion['Normal'] = fuzz.trimf(argumentacion.universe, [2.5, 5, 7.5])
argumentacion['Buena'] = fuzz.trapmf(argumentacion.universe, [5, 7.5, float("inf") ,float("inf")])

#Negociación
negociacion = ctrl.Consequent(np.arange(0,10, 0.01), 'negociacion')
negociacion['Baja'] = fuzz.trapmf(negociacion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
negociacion['Media'] = fuzz.trimf(negociacion.universe, [2.5, 5, 7.5])
negociacion['Alta'] = fuzz.trapmf(negociacion.universe, [5, 7.5, float("inf") ,float("inf")])

#Adecuación al tema
adecuacion_al_tema = ctrl.Antecedent(np.arange(0,10, 0.01), 'adecuacion_al_tema')
adecuacion_al_tema['Baja'] = fuzz.trapmf(adecuacion_al_tema.universe, [float("-inf") ,float("-inf"), 2.5, 5])
adecuacion_al_tema['Media'] = fuzz.trimf(adecuacion_al_tema.universe, [2.5, 5, 7.5])
adecuacion_al_tema['Alta'] = fuzz.trapmf(adecuacion_al_tema.universe, [5, 7.5, float("inf") ,float("inf")])

"""#Concrecion
concrecion = ctrl.Antecedent(np.arange(0,10, 0.01), 'concrecion')
concrecion['Baja'] = fuzz.trapmf(concrecion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
concrecion['Media'] = fuzz.trimf(concrecion.universe, [2.5, 5, 7.5])
concrecion['Alta'] = fuzz.trapmf(concrecion.universe, [5, 7.5, float("inf") ,float("inf")])"""

#Constancia_expresion_ideas
constancia = ctrl.Antecedent(np.arange(0,10, 0.01), 'constancia')
constancia['Baja'] = fuzz.trapmf(constancia.universe, [float("-inf") ,float("-inf"), 2.5, 5])
constancia['Media'] = fuzz.trimf(constancia.universe, [2.5, 5, 7.5])
constancia['Alta'] = fuzz.trapmf(constancia.universe, [5, 7.5, float("inf") ,float("inf")])

#Ruido
ruido = ctrl.Antecedent(np.arange(0,1, 0.01), 'ruido')
ruido['Bajo'] = fuzz.trapmf(ruido.universe, [float("-inf") ,float("-inf"), 0, 0.14])
ruido['Medio'] = fuzz.trimf(ruido.universe, [0, 0.14, 0.42])
ruido['Alto'] = fuzz.trapmf(ruido.universe, [0.14, 0.42, float("inf") ,float("inf")])

#Comunicación
comunicacion = ctrl.Consequent(np.arange(0,10 , 0.01), 'comunicacion')
comunicacion['Mala'] = fuzz.trapmf(comunicacion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
comunicacion['Media'] = fuzz.trimf(comunicacion.universe, [2.5, 5, 7.5])
comunicacion['Buena'] = fuzz.trapmf(comunicacion.universe, [5, 7.5, float("inf") ,float("inf")])

#Seguridad en el texto
seguridad_texto = ctrl.Consequent(np.arange(0,10 , 0.01), 'seguridad_texto')
seguridad_texto['Baja'] = fuzz.trapmf(seguridad_texto.universe, [float("-inf") ,float("-inf"), 2.5, 5])
seguridad_texto['Media'] = fuzz.trimf(seguridad_texto.universe, [2.5, 5, 7.5])
seguridad_texto['Alta'] = fuzz.trapmf(seguridad_texto.universe, [5, 7.5, float("inf") ,float("inf")])

#Seguridad audio
seguridad_audio = ctrl.Consequent(np.arange(0,10 , 0.01), 'seguridad_audio')
seguridad_audio['Baja'] = fuzz.trapmf(seguridad_audio.universe, [float("-inf") ,float("-inf"), 2.5, 5])
seguridad_audio['Media'] = fuzz.trimf(seguridad_audio.universe, [2.5, 5, 7.5])
seguridad_audio['Alta'] = fuzz.trapmf(seguridad_audio.universe, [5, 7.5, float("inf") ,float("inf")])

#Seguridad expresion
seguridad_expresion = ctrl.Consequent(np.arange(0,10 , 0.01), 'seguridad_expresion')
seguridad_expresion['Baja'] = fuzz.trapmf(seguridad_expresion.universe, [float("-inf") ,float("-inf"), 2.5, 5])
seguridad_expresion['Media'] = fuzz.trimf(seguridad_expresion.universe, [2.5, 5, 7.5])
seguridad_expresion['Alta'] = fuzz.trapmf(seguridad_expresion.universe, [5, 7.5, float("inf") ,float("inf")])

#Posición corporal
posicion_corporal = ctrl.Consequent(np.arange(0,10 , 0.01), 'posicion_corporal')
posicion_corporal['Mala'] = fuzz.trapmf(posicion_corporal.universe, [float("-inf") ,float("-inf"), 2.5, 5])
posicion_corporal['Media'] = fuzz.trimf(posicion_corporal.universe, [2.5, 5, 7.5])
posicion_corporal['Buena'] = fuzz.trapmf(posicion_corporal.universe, [5, 7.5, float("inf") ,float("inf")])

#Seguridad persona
seguridad_persona = ctrl.Consequent(np.arange(0,10 , 0.01), 'seguridad_persona')
seguridad_persona['Baja'] = fuzz.trapmf(seguridad_persona.universe, [float("-inf") ,float("-inf"), 2.5, 5])
seguridad_persona['Media'] = fuzz.trimf(seguridad_persona.universe, [2.5, 5, 7.5])
seguridad_persona['Alta'] = fuzz.trapmf(seguridad_persona.universe, [5, 7.5, float("inf") ,float("inf")])

#Autoestima
autoestima = ctrl.Consequent(np.arange(0,10 , 0.01), 'autoestima')
autoestima['Baja'] = fuzz.trapmf(autoestima.universe, [float("-inf") ,float("-inf"), 2.5, 5])
autoestima['Media'] = fuzz.trimf(autoestima.universe, [2.5, 5, 7.5])
autoestima['Alta'] = fuzz.trapmf(autoestima.universe, [5, 7.5, float("inf") ,float("inf")])

#Originalidad
originalidad = ctrl.Antecedent(np.arange(0,10 , 0.01), 'originalidad')
originalidad['Baja'] = fuzz.trapmf(originalidad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
originalidad['Media'] = fuzz.trimf(originalidad.universe, [2.5, 5, 7.5])
originalidad['Alta'] = fuzz.trapmf(originalidad.universe, [5, 7.5, float("inf") ,float("inf")])

#Congruencia
congruencia = ctrl.Antecedent(np.arange(0,10 , 0.01), 'congruencia')
congruencia['Baja'] = fuzz.trapmf(congruencia.universe, [float("-inf") ,float("-inf"), 2.5, 5])
congruencia['Media'] = fuzz.trimf(congruencia.universe, [2.5, 5, 7.5])
congruencia['Alta'] = fuzz.trapmf(congruencia.universe, [5, 7.5, float("inf") ,float("inf")])

#exactitud
exactitud = ctrl.Consequent(np.arange(0,10 , 0.01), 'exactitud')
exactitud['Baja'] = fuzz.trapmf(exactitud.universe, [float("-inf") ,float("-inf"), 2.5, 5])
exactitud['Media'] = fuzz.trimf(exactitud.universe, [2.5, 5, 7.5])
exactitud['Alta'] = fuzz.trapmf(exactitud.universe, [5, 7.5, float("inf") ,float("inf")])

#Concisión liderazgo
concision_liderazgo = ctrl.Consequent(np.arange(0,10 , 0.01), 'concision_liderazgo')
concision_liderazgo['Baja'] = fuzz.trapmf(concision_liderazgo.universe, [float("-inf") ,float("-inf"), 2.5, 5])
concision_liderazgo['Media'] = fuzz.trimf(concision_liderazgo.universe, [2.5, 5, 7.5])
concision_liderazgo['Alta'] = fuzz.trapmf(concision_liderazgo.universe, [5, 7.5, float("inf") ,float("inf")])

#Serenidad
serenidad = ctrl.Antecedent(np.arange(0,10 , 0.01), 'serenidad')
serenidad['Baja'] = fuzz.trapmf(serenidad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
serenidad['Media'] = fuzz.trimf(serenidad.universe, [2.5, 5, 7.5])
serenidad['Alta'] = fuzz.trapmf(serenidad.universe, [5, 7.5, float("inf") ,float("inf")])

#uniformidad_voz
uniformidad_voz = ctrl.Antecedent(np.arange(0,10 , 0.01), 'uniformidad_voz')
uniformidad_voz['Baja'] = fuzz.trapmf(uniformidad_voz.universe, [float("-inf") ,float("-inf"), 2.5, 5])
uniformidad_voz['Media'] = fuzz.trimf(uniformidad_voz.universe, [2.5, 5, 7.5])
uniformidad_voz['Alta'] = fuzz.trapmf(uniformidad_voz.universe, [5, 7.5, float("inf") ,float("inf")])

#Seguridad liderazgo
seguridad_liderazgo = ctrl.Consequent(np.arange(0,10, 0.01), 'seguridad_liderazgo')
seguridad_liderazgo['Baja'] = fuzz.trapmf(seguridad_liderazgo.universe, [float("-inf") ,float("-inf"), 2.5, 5])
seguridad_liderazgo['Media'] = fuzz.trimf(seguridad_liderazgo.universe, [2.5, 5, 7.5])
seguridad_liderazgo['Alta'] = fuzz.trapmf(seguridad_liderazgo.universe, [5, 7.5, float("inf") ,float("inf")])

"""#Respeto
respeto = ctrl.Antecedent(np.arange(0,10 , 0.01), 'respeto')
respeto['Poco'] = fuzz.trapmf(respeto.universe, [0,0, 2.5, 5])
respeto['Mucho'] = fuzz.trapmf(respeto.universe, [2.5, 5, 10, 10])"""

#Argumentacion liderazgo
argumentacion_liderazgo = ctrl.Antecedent(np.arange(0,10, 0.01), 'argumentacion_liderazgo')
argumentacion_liderazgo['Mala'] = fuzz.trapmf(argumentacion_liderazgo.universe, [float("-inf") ,float("-inf"), 2.5, 5])
argumentacion_liderazgo['Normal'] = fuzz.trimf(argumentacion_liderazgo.universe, [2.5, 5, 7.5])
argumentacion_liderazgo['Buena'] = fuzz.trapmf(argumentacion_liderazgo.universe, [5, 7.5, float("inf") ,float("inf")])

#liderazgo
liderazgo = ctrl.Consequent(np.arange(0,10, 0.01), 'liderazgo')
liderazgo['Malo'] = fuzz.trapmf(liderazgo.universe, [float("-inf") ,float("-inf"), 2.5, 5])
liderazgo['Normal'] = fuzz.trimf(liderazgo.universe, [2.5, 5, 7.5])
liderazgo['Bueno'] = fuzz.trapmf(liderazgo.universe, [5, 7.5, float("inf") ,float("inf")])

#lenguaje no verbal
lenguaje_no_verbal = ctrl.Antecedent(np.arange(0,10 , 0.01), 'lenguaje_no_verbal')
lenguaje_no_verbal['Malo'] = fuzz.trapmf(lenguaje_no_verbal.universe, [float("-inf") ,float("-inf"), 2.5, 5])
lenguaje_no_verbal['Normal'] = fuzz.trimf(lenguaje_no_verbal.universe, [2.5, 5, 7.5])
lenguaje_no_verbal['Bueno'] = fuzz.trapmf(lenguaje_no_verbal.universe, [5, 7.5, float("inf") ,float("inf")])

#control_estres
control_estres = ctrl.Consequent(np.arange(0,10, 0.01), 'control_estres')
control_estres['Bajo'] = fuzz.trapmf(control_estres.universe, [float("-inf") ,float("-inf"), 2.5, 5])
control_estres['Medio'] = fuzz.trimf(control_estres.universe, [2.5, 5, 7.5])
control_estres['Alto'] = fuzz.trapmf(control_estres.universe, [5, 7.5, float("inf") ,float("inf")])

#Vocabulario
vocabulario = ctrl.Antecedent(np.arange(0,10 , 0.01), 'vocabulario')
vocabulario['Malo'] = fuzz.trapmf(vocabulario.universe, [float("-inf") ,float("-inf"), 2.5, 5])
vocabulario['Normal'] = fuzz.trimf(vocabulario.universe, [2.5, 5, 7.5])
vocabulario['Bueno'] = fuzz.trapmf(vocabulario.universe, [5, 7.5, float("inf") ,float("inf")])

#Ideas
ideas = ctrl.Antecedent(np.arange(0,10 , 0.01), 'ideas')
ideas['Malo'] = fuzz.trapmf(ideas.universe, [float("-inf") ,float("-inf"), 2.5, 5])
ideas['Normal'] = fuzz.trimf(ideas.universe, [2.5, 5, 7.5])
ideas['Bueno'] = fuzz.trapmf(ideas.universe, [5, 7.5, float("inf") ,float("inf")])

#Calidad texto
calidad_texto = ctrl.Consequent(np.arange(0,10 , 0.01), 'calidad_texto')
calidad_texto['Baja'] = fuzz.trapmf(calidad_texto.universe, [float("-inf") ,float("-inf"), 2.5, 5])
calidad_texto['Media'] = fuzz.trimf(calidad_texto.universe, [2.5, 5, 7.5])
calidad_texto['Alta'] = fuzz.trapmf(calidad_texto.universe, [5, 7.5, float("inf") ,float("inf")])

#Creatividad
creatividad = ctrl.Consequent(np.arange(0,10, 0.01), 'creatividad')
creatividad['Baja'] = fuzz.trapmf(creatividad.universe, [float("-inf") ,float("-inf"), 2.5, 5])
creatividad['Media'] = fuzz.trimf(creatividad.universe, [2.5, 5, 7.5])
creatividad['Alta'] = fuzz.trapmf(creatividad.universe, [5, 7.5, float("inf") ,float("inf")])



def lectura_reglas_borrosas(df_reglas):
    reglas = []
    detalles_reglas = []
    

    for index,row in df_reglas.iterrows():
      try:
        antecedente = row['antecedente']
        consecuente = row['consecuente']
        v_consecuente = row['consecuente_valor']
        

        #Remover IF
        antecedente = antecedente[3:]

        #Remover [ ], AND y OR
        antecedente = antecedente.replace("[","[\'")
        antecedente = antecedente.replace("]","\']")
        antecedente = antecedente.replace("AND", "&")
        antecedente = antecedente.replace("OR", "|")

        #Consecuente
        consecuente = consecuente+"[\'"+v_consecuente+"\']"
        detalles_reglas=ctrl.Rule(eval(antecedente), eval(consecuente))
        reglas.append(detalles_reglas)
      except Exception as e:
        
        print("Exception in rule: ", row, e)
    return reglas

"""#Para preevaluacion
reglas_preevaluacion = lectura_reglas_borrosas(os.path.join(dictionaries_folder, 'reglas_preevaluacion.csv'))
preevaluacion_ctrl = ctrl.ControlSystem(reglas_preevaluacion)
preevaluacion_c = ctrl.ControlSystemSimulation(preevaluacion_ctrl)


def razonamiento_preevaluacion(v_duracion, v_dimension_video):
      preevaluacion_c.input['duracion'] = v_duracion
      preevaluacion_c.input['dimension_video'] = v_dimension_video
      preevaluacion_c.compute()
      
      v_calidad_video = round(preevaluacion_c.output['calidad_video'],2)
      etiqueta_calidad_video = etiqueta_linguistica(v_calidad_video, calidad_video)

      #Obtener las etiquetas de los antecedentes. Importante para justificar en caso de que el video no pase
      etiqueta_duracion = etiqueta_linguistica(v_duracion, duracion)
      etiqueta_dimension_video = etiqueta_linguistica(v_dimension_video, dimension_video)

      return {'calidad_video':v_calidad_video, 'etiqueta_calidad':etiqueta_calidad_video,
              'etiqueta_duracion':etiqueta_duracion, 'etiqueta_dimension_video':etiqueta_dimension_video}"""

#Para evaluacion
"""reglas_evaluacion = lectura_reglas_borrosas(os.path.join(dictionaries_folder, 'reglas_evaluacion_nuevas.csv'))
evaluacion_ctrl = ctrl.ControlSystem(reglas_evaluacion)
evaluacion_c = ctrl.ControlSystemSimulation(evaluacion_ctrl)"""

#Definición de la colección de las reglas
reglas_evaluacion_collection = evaluacion_soft_skills_db['rules_evaluation']
reglas_soft_skills = list(reglas_evaluacion_collection.find({},{'_id':0}))
#Conversión de las reglas a un dataframe
df_reglas_soft_skills = pd.DataFrame(reglas_soft_skills)
reglas_evaluacion = lectura_reglas_borrosas(df_reglas_soft_skills)
evaluacion_ctrl = ctrl.ControlSystem(reglas_evaluacion)
evaluacion_c = ctrl.ControlSystemSimulation(evaluacion_ctrl)

def razonamiento_evaluacion(metricas_p_inferencia, person_folder):
    moods_r = ['Reading', 'Normal', 'Passionately', 'Silent']
    
    v_tiempo_de_reaccion = metricas_p_inferencia['tiempo_de_reaccion']
    evaluacion_c.input['tiempo_de_reaccion'] = v_tiempo_de_reaccion
    v_fluidez = metricas_p_inferencia['fluidez']
    evaluacion_c.input['fluidez'] = v_fluidez
    v_rapidez_del_habla = metricas_p_inferencia['rapidez_del_habla']
    evaluacion_c.input['rapidez_del_habla'] = v_rapidez_del_habla
    v_organizacion = metricas_p_inferencia['organizacion']
    evaluacion_c.input['organizacion'] = v_organizacion

    v_mood = moods_r.index(metricas_p_inferencia['mood'])+1

    evaluacion_c.input['mood'] = v_mood
    v_vaguedad = metricas_p_inferencia['vaguedad']
    evaluacion_c.input['vaguedad'] = v_vaguedad
    v_ejemplos = metricas_p_inferencia['ejemplos']
    evaluacion_c.input['ejemplos'] = v_ejemplos
    v_muletillas = metricas_p_inferencia['muletillas']
    evaluacion_c.input['muletillas'] = v_muletillas
    v_sonrisa = metricas_p_inferencia['sonrisas']
    evaluacion_c.input['sonrisa'] = v_sonrisa

    v_expresion = metricas_p_inferencia['expresion']

    evaluacion_c.input['expresion'] = v_expresion
    v_constancia = metricas_p_inferencia['constancia']
    evaluacion_c.input['constancia'] = v_constancia
    v_ruido = metricas_p_inferencia['ruido']
    evaluacion_c.input['ruido'] = v_ruido
    v_adecuacion_al_tema = metricas_p_inferencia['adecuacion_al_tema']
    evaluacion_c.input['adecuacion_al_tema'] = v_adecuacion_al_tema
    v_cambios_de_postura = metricas_p_inferencia['cambios_de_postura']
    evaluacion_c.input['cambios_de_postura'] = v_cambios_de_postura
    v_originalidad = metricas_p_inferencia['originalidad']
    evaluacion_c.input['originalidad'] = v_originalidad
    v_congruencia = metricas_p_inferencia['congruencia']
    evaluacion_c.input['congruencia'] = v_congruencia
    v_densidad = metricas_p_inferencia['densidad']
    evaluacion_c.input['densidad'] = v_densidad
    v_serenidad = metricas_p_inferencia['serenidad']
    evaluacion_c.input['serenidad'] = v_serenidad
    v_uniformidad_voz = metricas_p_inferencia['uniformidad_voz']
    evaluacion_c.input['uniformidad_voz'] = v_uniformidad_voz
    #evaluacion_c.input['respeto'] = metricas_p_inferencia['respeto']

    v_vocabulario = metricas_p_inferencia['vocabulario']
    evaluacion_c.input['vocabulario'] = v_vocabulario

    v_ideas = metricas_p_inferencia['ideas']
    evaluacion_c.input['ideas'] = v_ideas
    v_argumentacion_liderazgo = metricas_p_inferencia['argumentacion_liderazgo']
    evaluacion_c.input['argumentacion_liderazgo'] = v_argumentacion_liderazgo

    #Cálculo del valor de la mirada
    v_mirada = metricas_p_inferencia['mirada']
    salida_mirada = fuzz.interp_membership(np.arange(0, 10, 0.01), mirada['Centrada'].mf, v_mirada)
    if salida_mirada>0.5 and v_mood==1:
      v_mirada = v_mirada*0.5
      print("Check it, the person read with focused gaze")
    evaluacion_c.input['mirada'] = v_mirada

    #Cálculo de lenguale_no_verbal (versión 1) Eliminar después
    """
    v_peso_mood = metricas_p_inferencia['peso_mood']
    v_parpadeo = metricas_p_inferencia['parpadeos']
    v_mirada_furtiva = metricas_p_inferencia['mirada_furtiva']
    v_emocion_estres = metricas_p_inferencia['emocion_estres']
    v_lenguaje_no_verbal_a = 0

    #Si la persona está leyendo...
    if v_peso_mood==3:
      v_lenguaje_no_verbal_a = 1.5
      
    else:
      #Si no está leyendo pero parpadea frecuentemente 
      if v_parpadeo>6.5:
        v_lenguaje_no_verbal_a += 0
        
      else:
        #Si el índice de parpadeos es mayor que 2,5 y menor que 6,5, añadir 3 puntos al lenguaje corporal
        if v_parpadeo>2.5:
          v_lenguaje_no_verbal_a+=3
        #Si el índice de parpadeos es hasta 2,5, añadir 1,5 puntos al lenguaje corporal
        else:
          v_lenguaje_no_verbal_a+=1.5
          
      
      
      if v_mirada_furtiva>5:
        if v_mirada_furtiva>9:
          v_lenguaje_no_verbal_a+=1
          
        else:
          v_lenguaje_no_verbal_a+=0
          
      else:
        if v_mirada_furtiva<2.5:
          v_lenguaje_no_verbal_a+=2
          
        else:
          v_lenguaje_no_verbal_a+=1
          
      if v_mirada_furtiva>6:
        v_lenguaje_no_verbal_a+=0
        
      else:
        if v_mirada_furtiva<2:
          v_lenguaje_no_verbal_a+=2
          
        else:
          v_lenguaje_no_verbal_a+=1
        
      
      if v_emocion_estres>8:
        v_lenguaje_no_verbal_a+=0
        
      else:
        if v_emocion_estres<3:
          v_lenguaje_no_verbal_a+=1
          
        else:
          v_lenguaje_no_verbal_a+=0.5"""

    #Lenguaje no verbal - nuevo
    v_lenguaje_no_verbal = metricas_p_inferencia['lenguaje_no_verbal']
    evaluacion_c.input['lenguaje_no_verbal'] = v_lenguaje_no_verbal
    """
    #Almacenar el lenguaje no verbal viejo y nuevo
    lenguaje_no_verbal_comparacion = {'anterior':v_lenguaje_no_verbal_a, 'nuevo':v_lenguaje_no_verbal}
    with open(person_folder+'lenguaje_no_verbal_comparacion.json', 'w') as outfile:
      json.dump(lenguaje_no_verbal_comparacion, outfile)  """
    evaluacion_c.compute()


    #Obtenemos las etiquetas linguisticas de los antecedentes
    etiqueta_argumentacion_liderazgo = etiqueta_linguistica(v_argumentacion_liderazgo, argumentacion_liderazgo)
    etiqueta_en_argumentacion_liderazgo = etiqueta_en(etiqueta_argumentacion_liderazgo)
    etiqueta_organizacion = etiqueta_linguistica(v_organizacion, organizacion)
    etiqueta_en_organizacion = etiqueta_en(etiqueta_organizacion)
    etiqueta_rapidez_del_habla = etiqueta_linguistica(v_rapidez_del_habla, rapidez_del_habla)
    etiqueta_en_rapidez_del_habla = etiqueta_en(etiqueta_rapidez_del_habla)
    etiqueta_fluidez = etiqueta_linguistica(v_fluidez, fluidez)
    etiqueta_en_fluidez = etiqueta_en(etiqueta_fluidez)
    etiqueta_tiempo_de_reaccion = etiqueta_linguistica(v_tiempo_de_reaccion, tiempo_de_reaccion)
    etiqueta_en_tiempo_de_reaccion = etiqueta_en(etiqueta_tiempo_de_reaccion)
    etiqueta_ejemplos = etiqueta_linguistica(v_ejemplos, ejemplos)
    etiqueta_en_ejemplos = etiqueta_en(etiqueta_ejemplos)
    etiqueta_vaguedad = etiqueta_linguistica(v_vaguedad, vaguedad)
    etiqueta_en_vaguedad = etiqueta_en(etiqueta_vaguedad)
    etiqueta_densidad = etiqueta_linguistica(v_densidad, densidad)
    etiqueta_en_densidad = etiqueta_en(etiqueta_densidad)
    etiqueta_originalidad = etiqueta_linguistica(v_originalidad, originalidad)
    etiqueta_en_originalidad = etiqueta_en(etiqueta_originalidad)
    etiqueta_congruencia = etiqueta_linguistica(v_congruencia, congruencia)
    etiqueta_en_congruencia = etiqueta_en(etiqueta_congruencia)
    etiqueta_mirada = etiqueta_linguistica(v_mirada, mirada)
    etiqueta_en_mirada = etiqueta_en(etiqueta_mirada)
    etiqueta_uniformidad_voz = etiqueta_linguistica(v_uniformidad_voz, uniformidad_voz)
    etiqueta_en_uniformidad_voz = etiqueta_en(etiqueta_uniformidad_voz)
    etiqueta_serenidad = etiqueta_linguistica(v_serenidad, serenidad)
    etiqueta_en_serenidad = etiqueta_en(etiqueta_serenidad)
    etiqueta_muletillas = etiqueta_linguistica(v_muletillas, muletillas)
    etiqueta_en_muletillas = etiqueta_en(etiqueta_muletillas)
    etiqueta_constancia = etiqueta_linguistica(v_constancia, constancia)
    etiqueta_en_constancia = etiqueta_en(etiqueta_constancia)
    etiqueta_ruido = etiqueta_linguistica(v_ruido, ruido)
    etiqueta_en_ruido = etiqueta_en(etiqueta_ruido)
    etiqueta_vocabulario = etiqueta_linguistica(v_vocabulario, vocabulario)
    etiqueta_en_vocabulario = etiqueta_en(etiqueta_vocabulario)
    etiqueta_ideas = etiqueta_linguistica(v_ideas, ideas)
    etiqueta_en_ideas = etiqueta_en(etiqueta_ideas)
    etiqueta_adecuacion_al_tema = etiqueta_linguistica(v_adecuacion_al_tema, adecuacion_al_tema)
    etiqueta_en_adecuacion_al_tema = etiqueta_en(etiqueta_adecuacion_al_tema)
    etiqueta_cambios_de_postura = etiqueta_linguistica(v_cambios_de_postura, cambios_de_postura)
    etiqueta_en_cambios_de_postura = etiqueta_en(etiqueta_cambios_de_postura)
    etiqueta_sonrisa = etiqueta_linguistica(v_sonrisa, sonrisa)
    etiqueta_en_sonrisa = etiqueta_en(etiqueta_sonrisa)
    etiqueta_lenguaje_no_verbal = etiqueta_linguistica(v_lenguaje_no_verbal, lenguaje_no_verbal)
    etiqueta_en_lenguaje_no_verbal = etiqueta_en(etiqueta_lenguaje_no_verbal)
    etiqueta_expresion = etiqueta_linguistica(v_expresion, expresion)
    etiqueta_en_expresion = etiqueta_en(etiqueta_expresion)
    etiquetas_antecedentes = [
      {'antecedente': 'tiempo_de_reaccion', 'etiqueta': etiqueta_tiempo_de_reaccion},
      {'antecedente':'fluidez', 'etiqueta': etiqueta_fluidez},
      {'antecedente':'rapidez_del_habla', 'etiqueta': etiqueta_rapidez_del_habla},
      {'antecedente':'organizacion', 'etiqueta': etiqueta_organizacion},
      {'antecedente':'mood', 'etiqueta': etiqueta_linguistica(v_mood, mood)},
      {'antecedente':'vaguedad', 'etiqueta': etiqueta_vaguedad},
      {'antecedente':'ejemplos', 'etiqueta': etiqueta_ejemplos},
      {'antecedente':'muletillas', 'etiqueta': etiqueta_muletillas},
      {'antecedente':'densidad', 'etiqueta': etiqueta_densidad},
      {'antecedente':'sonrisa','etiqueta': etiqueta_sonrisa},
      {'antecedente':'mirada', 'etiqueta': etiqueta_en_mirada},
      {'antecedente':'cambios_de_postura', 'etiqueta': etiqueta_cambios_de_postura},
      {'antecedente':'expresion','etiqueta': etiqueta_expresion},
      {'antecedente':'adecuacion_al_tema', 'etiqueta': etiqueta_adecuacion_al_tema},
      {'antecedente':'constancia','etiqueta': etiqueta_constancia},
      {'antecedente':'ruido', 'etiqueta': etiqueta_ruido},
      {'antecedente':'originalidad', 'etiqueta': etiqueta_originalidad},
      {'antecedente':'congruencia','etiqueta': etiqueta_congruencia},
      {'antecedente':'serenidad', 'etiqueta': etiqueta_serenidad},
      {'antecedente':'uniformidad_voz', 'etiqueta': etiqueta_uniformidad_voz},
      {'antecedente':'lenguaje_no_verbal', 'etiqueta': etiqueta_lenguaje_no_verbal},
      {'antecedente':'vocabulario', 'etiqueta': etiqueta_vocabulario},
      {'antecedente':'ideas', 'etiqueta': etiqueta_ideas},
      {'antecedente':'argumentacion_liderazgo', 'etiqueta': etiqueta_argumentacion_liderazgo}
    ]

    v_velocidad = round(evaluacion_c.output['velocidad'],2)
    etiqueta_velocidad = etiqueta_linguistica(v_velocidad, velocidad)
    etiqueta_en_velocidad = etiqueta_en(etiqueta_velocidad)
    v_firmeza = round(evaluacion_c.output['firmeza'],2)
    etiqueta_firmeza = etiqueta_linguistica(v_firmeza, firmeza)
    etiqueta_en_firmeza = etiqueta_en(etiqueta_firmeza)
    v_concision = round(evaluacion_c.output['concision'],2)
    etiqueta_concision = etiqueta_linguistica(v_concision, concision)
    etiqueta_en_concision = etiqueta_en(etiqueta_concision)
    v_claridad = round(evaluacion_c.output['claridad'],2)
    etiqueta_claridad = etiqueta_linguistica(v_claridad, claridad)
    etiqueta_en_claridad = etiqueta_en(etiqueta_claridad)
    v_decision = round(evaluacion_c.output['decision_making'],2)
    etiqueta_dm = etiqueta_linguistica(v_decision, decision_making)
    etiqueta_en_dm = etiqueta_en(etiqueta_dm)

    v_empatia = round(evaluacion_c.output['empatia'],2)
    etiqueta_empatia = etiqueta_linguistica(v_empatia, empatia)
    etiqueta_en_empatia = etiqueta_en(etiqueta_empatia)
    v_argumentacion = round(evaluacion_c.output['argumentacion'], 2)
    etiqueta_argumentacion = etiqueta_linguistica(v_argumentacion, argumentacion)
    etiqueta_en_argumentacion = etiqueta_en(etiqueta_argumentacion)
    v_calidad_argumentacion = round(evaluacion_c.output['calidad_argumentacion'],2)
    etiqueta_calidad_argumentacion = etiqueta_linguistica(v_calidad_argumentacion, calidad_argumentacion)
    etiqueta_en_calidad_argumentacion = etiqueta_en(etiqueta_calidad_argumentacion)
    v_negociacion = round(evaluacion_c.output['negociacion'],2)
    etiqueta_negociacion = etiqueta_linguistica(v_negociacion, negociacion)
    etiqueta_en_negociacion = etiqueta_en(etiqueta_negociacion)

    #Resolución de conflictos
    v_resolucion_conflictos = round((((v_calidad_argumentacion + v_expresion) / 2) + v_empatia)/2,2)

    #Comunicación [0, 10]
    v_comunicacion = round(evaluacion_c.output['comunicacion'],2)
    etiqueta_comunicacion = etiqueta_linguistica(v_comunicacion, comunicacion)
    etiqueta_en_comunicacion = etiqueta_en(etiqueta_comunicacion)
    v_seguridad_texto = round(evaluacion_c.output['seguridad_texto'],2)
    etiqueta_seguridad_texto = etiqueta_linguistica(v_seguridad_texto, seguridad_texto)
    etiqueta_en_seguridad_texto = etiqueta_en(etiqueta_seguridad_texto)
    v_seguridad_audio = round(evaluacion_c.output['seguridad_audio'],2)
    etiqueta_seguridad_audio = etiqueta_linguistica(v_seguridad_audio, seguridad_audio)
    etiqueta_en_seguridad_audio = etiqueta_en(etiqueta_seguridad_audio)
    v_posicion_corporal = round(evaluacion_c.output['posicion_corporal'],2)
    etiqueta_posicion_corporal = etiqueta_linguistica(v_posicion_corporal, posicion_corporal)
    etiqueta_en_posicion_corporal = etiqueta_en(etiqueta_posicion_corporal)

    v_exactitud = round(evaluacion_c.output['exactitud'],2)
    etiqueta_exactitud = etiqueta_linguistica(v_exactitud, exactitud)
    etiqueta_en_exactitud = etiqueta_en(etiqueta_exactitud)
    v_concision_liderazgo = round(evaluacion_c.output['concision_liderazgo'],2)
    etiqueta_concision_liderazgo = etiqueta_linguistica(v_concision_liderazgo, concision_liderazgo)
    etiqueta_en_concision_liderazgo = etiqueta_en(etiqueta_concision_liderazgo)
    v_seguridad_liderazgo = round(evaluacion_c.output['seguridad_liderazgo'],2)
    etiqueta_seguridad_liderazgo = etiqueta_linguistica(v_seguridad_liderazgo, seguridad_liderazgo)
    etiqueta_en_seguridad_liderazgo = etiqueta_en(etiqueta_seguridad_liderazgo)
    #v_argumentacion_liderazgo = (round(evaluacion_c.output['argumentacion_liderazgo'],2))
    #etiqueta_argumentacion_liderazgo = etiqueta_linguistica(v_argumentacion_liderazgo, argumentacion_liderazgo)
    v_liderazgo = round(evaluacion_c.output['liderazgo'],2)
    etiqueta_liderazgo = etiqueta_linguistica(v_liderazgo, liderazgo)
    etiqueta_en_liderazgo = etiqueta_en(etiqueta_liderazgo)
    v_autoestima = round(evaluacion_c.output['autoestima'],2)
    etiqueta_autoestima  = etiqueta_linguistica(v_autoestima, autoestima)
    etiqueta_en_autoestima = etiqueta_en(etiqueta_autoestima)
    v_control_estres = round(evaluacion_c.output['control_estres'],2)
    etiqueta_control_estres = etiqueta_linguistica(v_control_estres, control_estres)
    etiqueta_en_control_estres = etiqueta_en(etiqueta_control_estres)
    v_calidad_texto = round(evaluacion_c.output['calidad_texto'],2)
    etiqueta_calidad_texto = etiqueta_linguistica(v_calidad_texto, calidad_texto)
    etiqueta_en_calidad_texto = etiqueta_en(etiqueta_calidad_texto)
    v_creatividad = round(evaluacion_c.output['creatividad'],2)
    etiqueta_creatividad = etiqueta_linguistica(v_creatividad, creatividad)
    etiqueta_en_creatividad = etiqueta_en(etiqueta_creatividad)
    v_seguridad_persona = round(evaluacion_c.output['seguridad_persona'],2)
    etiqueta_seguridad_persona = etiqueta_linguistica(v_seguridad_persona, seguridad_persona)
    etiqueta_en_seguridad_persona = etiqueta_en(etiqueta_seguridad_persona)
    v_seguridad_expresion = round(evaluacion_c.output['seguridad_expresion'],2)
    etiqueta_seguridad_expresion = etiqueta_linguistica(v_seguridad_expresion, seguridad_expresion)
    etiqueta_en_seguridad_expresion = etiqueta_en(etiqueta_seguridad_expresion)
   

    resultado_final_total = {
        'decision_making':{'label':etiqueta_en_dm, 'score':v_decision, 'details':[{'conciseness':{'label':etiqueta_en_concision, 'score':v_concision, 'details':[{'firmness':{'label':etiqueta_en_firmeza, 'score':v_firmeza, 'details':[{'organization':{'label':etiqueta_en_organizacion, 'score':v_organizacion}},
                                                                                                                                                                    {'mood':{'label':moods_r[v_mood-1], 'score':v_mood}}]}},
                                                                                               {'speed':{'label':etiqueta_en_velocidad, 'score':v_velocidad, 'details':[{'speech_speed':{'label':etiqueta_en_rapidez_del_habla, 'score':v_rapidez_del_habla}},
                                                                                                                                                                     {'fluency':{'label':etiqueta_en_fluidez, 'score':v_fluidez}},
                                                                                                                                                                     {'reaction_time':{'label':etiqueta_en_tiempo_de_reaccion, 'score':v_tiempo_de_reaccion}}]}}]}},
                   {'clarity':{'label':etiqueta_en_claridad, 'score':v_claridad, 'details':[{'examples':{'label':etiqueta_en_ejemplos, 'score':v_ejemplos}},
                                                                                            {'vagueness':{'label':etiqueta_en_vaguedad, 'score':v_vaguedad}}]}}]},

        'negotiation':{'label':etiqueta_en_negociacion, 'score':v_negociacion, 'details':[{'argumentation':{'label':etiqueta_en_argumentacion, 'score':v_argumentacion, 'details':[{'expression':{'label':etiqueta_en_expresion, 'score':v_expresion}},
                                                                                                                                                                                   {'argument_quality':{'label':etiqueta_en_calidad_argumentacion, 'score':v_calidad_argumentacion, 'details':[{'vagueness':{'label':etiqueta_en_vaguedad, 'score':v_vaguedad}},
                                                                                                                                                                                                                                                                                               {'fluency':{'label':etiqueta_en_fluidez, 'score':v_fluidez}}]}}]}},
                                                                                          {'empathy':{'label':etiqueta_en_empatia, 'score':v_empatia, 'details':[{'gaze':{'label':etiqueta_en_mirada, 'score':v_mirada}},
                                                                                                                                                                 {'smile':{'label':etiqueta_en_sonrisa, 'score':v_sonrisa}}]}}]},

        'leadership':{'label':etiqueta_en_liderazgo, 'score':v_liderazgo, 'details':[{'conciseness_in_leadership':{'label':etiqueta_en_concision_liderazgo, 'score':v_concision_liderazgo, 'details':[{'speed':{'label':etiqueta_en_velocidad, 'score':v_velocidad, 'details':[{'speech_speed':{'label':etiqueta_en_rapidez_del_habla, 'score':v_rapidez_del_habla}},
                                                                                                                                                                                                                                                                               {'fluency':{'label':etiqueta_en_fluidez, 'score':v_fluidez}},
                                                                                                                                                                                                                                                                               {'reaction_time':{'label':etiqueta_en_tiempo_de_reaccion, 'score':v_tiempo_de_reaccion}}]}},
                                                                                                                                                                                                      {'density':{'label':etiqueta_en_densidad, 'score':v_densidad}},
                                                                                                                                                                                                      {'accuracy':{'label':etiqueta_en_exactitud, 'score':v_exactitud, 'details':[{'originality':{'label':etiqueta_en_originalidad, 'score':v_originalidad}},
                                                                                                                                                                                                                                                                                  {'congruence':{'label':etiqueta_en_congruencia, 'score':v_congruencia}}]}}]}},
                                                                                     {'security_in_leadership':{'label':etiqueta_en_seguridad_liderazgo, 'score':v_seguridad_liderazgo, 'details':[{'gaze':{'label':etiqueta_en_mirada, 'score':v_mirada}},
                                                                                                                                                                                                   {'voice_uniformity':{'label':etiqueta_en_uniformidad_voz, 'score':v_uniformidad_voz}},
                                                                                                                                                                                                   {'serenity':{'label':etiqueta_en_serenidad, 'score':v_serenidad}}]}},
                                                                                     {'argumentation_in_leadership':{'label':etiqueta_en_argumentacion_liderazgo, 'score':v_argumentacion_liderazgo}}]},

        'stress_control':{'label':etiqueta_en_control_estres, 'score':v_control_estres, 'details':[{'non_verbal_language':{'label':etiqueta_en_lenguaje_no_verbal, 'score':v_lenguaje_no_verbal}},
                                                                                                   {'communication':{'label':etiqueta_en_comunicacion, 'score':v_comunicacion, 'details':[{'fillers':{'label':etiqueta_en_muletillas, 'score':v_muletillas}},
                                                                                                                                                                                          {'consistency':{'label':etiqueta_en_constancia, 'score':v_constancia}},
                                                                                                                                                                                          {'noise':{'label':etiqueta_en_ruido, 'score':v_ruido}}]}},
                                                                                                   {'expression':{'label':etiqueta_en_expresion, 'score':v_expresion}}]},

        'creativity':{'label':etiqueta_en_creatividad, 'score':v_creatividad, 'details':[{'text_quality':{'label':etiqueta_en_calidad_texto, 'score':v_calidad_texto, 'details':[{'vocabulary':{'label':etiqueta_en_vocabulario, 'score':v_vocabulario}},
                                                                                                                                                                                 {'ideas':{'label':etiqueta_en_ideas, 'score':v_ideas}}]}},
                                                                                         {'non_verbal_language':{'label':etiqueta_en_lenguaje_no_verbal, 'score':v_lenguaje_no_verbal}}]},

        'self_esteem':{'label':etiqueta_en_autoestima, 'score':v_autoestima, 'details':[{'personal_security':{'label':etiqueta_en_seguridad_persona, 'score':v_seguridad_persona, 'details':[{'expression_security':{'label':etiqueta_en_seguridad_expresion, 'score':v_seguridad_expresion, 'details':[{'text_security':{'label':etiqueta_en_seguridad_texto, 'score':v_seguridad_texto, 'details':[{'topic_adequacy':{'labe':etiqueta_en_adecuacion_al_tema, 'score':v_adecuacion_al_tema}},
                                                                                                                                                                                                                                                                                                                                                                                                     {'vagueness':{'label':etiqueta_en_vaguedad, 'score':v_vaguedad}},
                                                                                                                                                                                                                                                                                                                                                                                                     {'fillers':{'label':etiqueta_en_muletillas, 'score':v_muletillas}}]}},
                                                                                                                                                                                                                                                                                                        {'audio_security':{'label':etiqueta_en_seguridad_audio, 'score':v_seguridad_audio, 'details':[{'reaction_time':{'label':etiqueta_en_tiempo_de_reaccion, 'score':v_tiempo_de_reaccion}},
                                                                                                                                                                                                                                                                                                                                                                                                      {'fluency':{'label':etiqueta_en_fluidez, 'score':v_fluidez}},
                                                                                                                                                                                                                                                                                                                                                                                                      {'speech_speed':{'label':etiqueta_en_rapidez_del_habla, 'score':v_rapidez_del_habla}}]}}]}},
                                                                                                                                                                                             {'body_position':{'label':etiqueta_en_posicion_corporal, 'score':v_posicion_corporal, 'details':[{'gaze':{'label':etiqueta_en_mirada, 'score':v_mirada}},
                                                                                                                                                                                                                                                                                              {'posture_changes':{'label':etiqueta_en_cambios_de_postura, 'score':v_cambios_de_postura}}]}}]}},
                                                                                        {'expression':{'label':etiqueta_en_expresion, 'score':v_expresion}}]}
    }
    return {'resultado_final':resultado_final_total, 'etiquetas_antecedentes':etiquetas_antecedentes}
                
    

def etiqueta_linguistica(valor, consecuente):
  dm_mval={}
  etiqueta = ""
  for t in consecuente.terms:
    
    val = np.interp(valor, consecuente.universe, consecuente[t].mf)
    dm_mval[t] = val
  
  #Ordenamos el grado de pertenencia de mayor a menor
  dm_mval = dict(sorted(dm_mval.items(), key=lambda item: item[1], reverse=True))

  for l in dm_mval:
    if dm_mval[l]>0.0:
      if etiqueta=="":
        etiqueta=l
      else:
        etiqueta = etiqueta+"/"+l

  #Versión 1, etiqueta con el valor más alto
  """max_dm_val = max(dm_mval.values())
  
  for l in dm_mval:
    if dm_mval[l]==max_dm_val:
      if etiqueta=="":
        etiqueta=l
      else:
        etiqueta = etiqueta+"/"+l
  #print(valor, etiqueta, max_dm_val,"\n")"""
  return etiqueta

def calculo_metricas(person_folder, person_id, tema):
  inicio_calculo_metricas = time.time()
  print('***Inference process***')
  resultado_control_borroso = os.path.join(person_folder, person_id+'_fuzzy_control_output.json')
  explicacion_de_skills = os.path.join(person_folder, person_id+'_soft_skills_explanation.txt')
  salida = []
  metricas_p_inferencia = os.path.join(person_folder, person_id+'_measures_for_inference.json')
  with open(metricas_p_inferencia, "r") as rf:
    decoded_data = json.load(rf)
  resultado_razonamiento = razonamiento_evaluacion(decoded_data, person_folder)
  resultado_metricas = resultado_razonamiento['resultado_final']
  etiquetas_antecedentes = resultado_razonamiento['etiquetas_antecedentes']
  

  fin_calculo_metricas = time.time()

  #Cálculo del grado de pertenencia de las reglas
  resultado_reglas = reglas_evaluacion(evaluacion_c)
 

  
  #Explicaciones
  explicacion_skills = generacion_explicaciones(resultado_reglas, etiquetas_antecedentes)

  #Traducimos al inglés las explicaciones generadas con el traductor de Google
  explicacion_skills = translator.translate(explicacion_skills, dest='en').text

  #Escribir las explicaciones en txt
  with open(explicacion_de_skills, "w") as wf:
    wf.write(explicacion_skills)
  

  #Generación del reporte usando las explicaciones como prompt.
  #Primero eliminamos los saltos de línea y los tabuladores
  explicacion_skills = explicacion_skills.replace('\n','')
  explicacion_skills = explicacion_skills.replace('\t','')
  #Generamos el reporte
  reporte_texto = reporte(explicacion_skills, '6 soft skills: decision making, negociation, leadership, stress control, creativity and self esteem', tema)
  
  #El reporte generado será añadido al fichero de métricas (resultado_metricas)
  try:
    resultado_metricas = {'id':person_id, 'report': reporte_texto, 'details':resultado_metricas}
    with open(resultado_control_borroso, "w") as outfile:
      json.dump(resultado_metricas, outfile, ensure_ascii= False)
    print("Soft skills evaluation completed")
  except Exception as e:
    print(e)

  duracion_calculo_metricas = round(fin_calculo_metricas - inicio_calculo_metricas,2)
  print('Inference process time:',duracion_calculo_metricas)
  return resultado_metricas



#Función para calcular el grado de pertenencia de las reglas del sistema borroso
def reglas_evaluacion(ctrl_sys_sim):
  #Crear una lista para almacenar los resultados de las reglas
  rules_details = []
  #Recibe un simulador de sistema de control
  rules_count = 0
  #Lee las reglas asociadas
  for rule in ctrl_sys_sim.ctrl.rules.all_rules:
    #Si se debe registrar, el valor de registrar cambia a True
    registrar=False
    rules_count+=1
    #Por cada regla, pasar el simulador de sistema de control
    rule_membership = rule.aggregate_firing[ctrl_sys_sim]
    #Detalles del resultado de la regla
    detalle_regla = {'Regla':'Regla '+str(rules_count),
                              'Antecedentes':rule.antecedent_terms,
                              'Consecuente':rule.consequent,
                              'Pertenencia':rule_membership}
    
      
    #Validar si el consecuente se ha guardado en la lista
    try:
      existente = next(item for item in rules_details if item['Consecuente'] == detalle_regla['Consecuente'])
     
      #Si el consecuente ya existe, verificar si el grado de pertenencia es mayor
      if existente['Pertenencia']<detalle_regla['Pertenencia']:
        #Eliminamos el consecuente existente con el grado de pertenencia menor
        rules_details.remove(existente)
        #Indicar que se debe almacenar
        registrar=True
    except:
      #Almacenar las reglas cuya pertenencia sea mayor a 0
      if detalle_regla['Pertenencia']>0:
        registrar=True
    #Si se debe registrar, almacenar el resultado de la regla
    if registrar==True:
        rules_details.append(detalle_regla)
  return rules_details
from pprint import pprint
def generacion_explicaciones(rules_details, etiquetas_antecedentes):
    explicacion_skills = ''
    etiqueta_mood = next(item for item in etiquetas_antecedentes if item['antecedente'] == 'mood')['etiqueta']
    skills_to_explain = ['decision_making','negociacion','liderazgo','control_estres', 'creatividad','autoestima']
    for skill in skills_to_explain:
      detalles_skill = reglas_consecuente(rules_details, skill, etiquetas_antecedentes)
      explicacion_skills+=explicacion_variable(0,-1,detalles_skill,etiqueta_mood)+'\n \n'
    return explicacion_skills

      

      
import string
alphabet = list(string.ascii_lowercase)
def explicacion_variable(orden, posicion,detalles_variable,etiqueta_mood):
  explicacion=''
  
  mood_explicacion = {'reading':'de lectura',
                             'normal':'normal',
                             'passionately':'apasionado'}
  
  
  if orden>0:
    for i in range(orden):
      explicacion +='\t'
  id_posicion=str(posicion)
  if posicion>-1:
    if orden==2:
      #Si es de orden 1, las viñetas son letras
      id_posicion = alphabet[posicion]
      
    else:
      posicion+=1
      if orden==3:
        #Si es de orden 3, las viñetas son números romanos
        id_posicion = intToRoman(posicion)
      else:
        id_posicion = str(posicion)
    explicacion+=id_posicion+') '
  try:
    
    #Obtenemos el nuevo nombre de la variable, es decir, con el artículo correspondiente
    n_variable = nuevo_nombre_variable(detalles_variable['tipo_nombre'], detalles_variable['nombre'])
    etiqueta_variable = detalles_variable['Etiqueta'].lower()
    #Si es el tono de voz, utilizar la equivalencia de descripción
    if detalles_variable['nombre_variable']=='mood':
      etiqueta_variable = mood_explicacion[etiqueta_variable]
    if 'p' in detalles_variable['tipo_nombre']:
      exp_variable = n_variable + ' son ' + etiqueta_variable+'. '
    else:
        if detalles_variable['nombre_variable']=='lenguaje_no_verbal':
            if etiqueta_mood=='Reading':
              exp_variable = n_variable + ' es ' + etiqueta_variable+', pues el tono de voz utilizado es de lectura y no se consideran los parpadeos, la mirada, los cambios de postura y los gestos.'
            else:
              exp_variable = n_variable + ' es ' + etiqueta_variable+' y se evalúa considerando los parpadeos, la mirada, los movimientos del rostro y los gestos.'
        else: 
          exp_variable = n_variable + ' es ' + etiqueta_variable+'. '
    exp_variable = exp_variable[0].upper() + exp_variable[1:]
    explicacion += exp_variable
    explicacion = explicacion[0].upper() + explicacion[1:] 
    if 'Antecedentes' in detalles_variable:
      dependencia_variable = n_variable + ' depende de '
      dependencia_variable = dependencia_variable[0].upper() + dependencia_variable[1:]
      explicacion+= dependencia_variable
      #Por cada antecedente de la variable, incluirlo en la explicación
      for antecedente in detalles_variable['Antecedentes']:
        
        #Obtenemos el nuevo nombre de la variable, es decir, con el artículo correspondiente
        n_antecedente = nuevo_nombre_variable(antecedente['tipo_nombre'], antecedente['nombre'])
        etiqueta_antecedente = antecedente['Etiqueta']
        #Si tiene solo un antecedente:
        if len(detalles_variable['Antecedentes'])==1:
          explicacion+= n_antecedente+'. \n'
        else:
          #Si es el último antecedente
          if antecedente==detalles_variable['Antecedentes'][-1]:
            explicacion+= 'y '+n_antecedente +'. \n'
          #Si es el penúltimo antecedente
          elif antecedente==detalles_variable['Antecedentes'][-2]:
            explicacion+= n_antecedente+' '
          else:
            explicacion+= n_antecedente+', '
      #Detalle de cada antecedente que es de tipo consecuente
      for antecedente in detalles_variable['Antecedentes']:
          posicion = detalles_variable['Antecedentes'].index(antecedente)
          explicacion += explicacion_variable(orden+1, posicion, antecedente,etiqueta_mood)

    else:
      explicacion+='\n'
  except Exception as e:
    print('Exception: ',e)

  #Correción de errores de ortografía
  explicacion = explicacion.replace('de el', 'del')
  #Mayúscula al inicio de la explicación 
  return explicacion


def nuevo_nombre_variable (tipo_nombre, nombre_variable):
  n_variable = ''
  if tipo_nombre=='f':
      n_variable = 'la ' +nombre_variable
  elif tipo_nombre=='m':
    n_variable = 'el '+nombre_variable
  elif tipo_nombre=='f,p':
    n_variable = 'las '+nombre_variable
  elif tipo_nombre=='m,p':
    n_variable = 'los '+nombre_variable
  else:
    n_variable = nombre_variable
  return n_variable

def reglas_consecuente(rules_details, consecuente, etiquetas_antecedentes):
  #reglas_consecuente = [item for item in rules_details if str(item['Consecuente'][0].term.parent).split('Consequent: ')[1] == consecuente]
  #return reglas_consecuente
  detalles_variable = next(item for item in descripcion_variables if item['nombre_variable'] == consecuente)
  if detalles_variable['tipo_variable']=='Consequent':
    #Buscamos las reglas que dan lugar al consecuente
    reglas_del_consecuente = [item for item in rules_details if str(item['Consecuente'][0].term.parent).split('Consequent: ')[1] == consecuente]
    #Ordenamos las reglas por el grado de pertenencia
    reglas_del_consecuente.sort(key=lambda x: x['Pertenencia'], reverse=True)
    reglas_variable = []
    etiqueta_variable = ''
    antecedentes_variable = []
    antecedentes_detalle = []
    #Definimos un contador de pertenencia
    contador_pertenencia = 0
    for regla in reglas_del_consecuente:
      #Si el contador de pertenencia es menor a 1, se puede agregar la regla
      if contador_pertenencia<1 and contador_pertenencia+regla['Pertenencia']<=1:
        #Agregamos la regla a la lista de reglas del consecuente
        reglas_variable.append(regla)
        #Incrementamos el contador de pertenencia
        contador_pertenencia+=regla['Pertenencia']
        #Registramos los antecedentes de la variable
        for antecedente in regla['Antecedentes']:
          nombre_antecedente = str(antecedente.parent).split(': ')[1]
          
          if nombre_antecedente not in antecedentes_variable:
            antecedentes_variable.append(nombre_antecedente)
            antecedentes_detalle.append(reglas_consecuente(rules_details, nombre_antecedente, etiquetas_antecedentes))
        if etiqueta_variable == '':
          etiqueta_variable = regla['Consecuente'][0].term.label
        elif etiqueta_variable!=regla['Consecuente'][0].term.label and regla['Consecuente'][0].term.label not in etiqueta_variable:
          etiqueta_variable +='/' + regla['Consecuente'][0].term.label
    detalles_variable['Reglas'] = reglas_variable
    detalles_variable['Etiqueta'] = etiqueta_variable
    detalles_variable['Antecedentes'] = antecedentes_detalle
  if detalles_variable['tipo_variable']=='Antecedent':
    #Buscamos la etiqueta en la lista de etiquetas de antecedentes
    try:
      etiqueta_variable = next(item for item in etiquetas_antecedentes if item['antecedente'] == detalles_variable['nombre_variable'])
      etiqueta_variable = etiqueta_variable['etiqueta']
      detalles_variable['Etiqueta'] = etiqueta_variable
    except Exception as e:
      print('Exception: ',e)
    
  return detalles_variable

#Función para convertir int a romano
def intToRoman(num):
  
    # Storing roman values of digits from 0-9
    # when placed at different places
    m = ["", "M", "MM", "MMM"]
    c = ["", "C", "CC", "CCC", "CD", "D",
         "DC", "DCC", "DCCC", "CM "]
    x = ["", "X", "XX", "XXX", "XL", "L",
         "LX", "LXX", "LXXX", "XC"]
    i = ["", "I", "II", "III", "IV", "V",
         "VI", "VII", "VIII", "IX"]
  
    # Converting to roman
    thousands = m[num // 1000]
    hundreds = c[(num % 1000) // 100]
    tens = x[(num % 100) // 10]
    ones = i[num % 10]
  
    ans = (thousands + hundreds +
           tens + ones)
  
    return ans

etiquetas_ingles = {'Low':['Bajo', 'Baja'],
                    'Medium':['Medio', 'Media'],
                    'High':['Alto', 'Alta'],
                    'Bad':['Malo', 'Mala'],
                    'Good':['Bueno', 'Buena'],
                    'Normal':['Normal'],
                    'Diffuse':['Difuso', 'Difusa'],
                    'Centered':['Centrado', 'Centrada'],
                    'Scarce':['Escaso', 'Escasa'],
                    'Frequent':['Frecuente'],
                    'Few':['Pocos', 'Pocas'],
                    'Regular':['Regular'],
                    'Many':['Muchos', 'Muchas']}

def etiqueta_en(etiqueta_es):
    # Separar la etiqueta en español si contiene una diagonal
    partes_es = etiqueta_es.split('/')
    partes_en = []
    
    # Traducir cada parte de la etiqueta
    for parte in partes_es:
        encontrado = False
        for etiqueta_en, etiquetas_es in etiquetas_ingles.items():
            if parte.strip() in etiquetas_es:
                partes_en.append(etiqueta_en)
                encontrado = True
                break
        # Si no se encuentra la traducción, se agrega tal cual
        if not encontrado:
            partes_en.append(parte.strip())
    
    # Unir las partes traducidas con una diagonal y devolver
    return '/'.join(partes_en)
    