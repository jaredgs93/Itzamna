#Importación de bibliotecas
from decouple import config
from openai import OpenAI

client = OpenAI(api_key = config('OPENAI_API_KEY'))

#Función para la generación del reporte
def translator(original_text):
    completion = client.chat.completions.create(
            messages = [{'role': 'system', 'content' : """Acts as a translator. You will receive a text and you must translate it into English"""},
                        {'role': 'user', 'content' : original_text}],
        
            model = 'gpt-3.5-turbo'
        )
    
        
    chat_response = completion.choices[0].message.content
    return chat_response