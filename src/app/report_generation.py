###SCRIPT FOR REPORT GENERATION USING A GPT MODEL FROM THE GENERATED PROMPT###

from decouple import config
from openai import OpenAI

client = OpenAI(api_key = config('OPENAI_API_KEY'))

# Function for report generation
def reporte(prompt, objetivo, tema):
    completion = client.chat.completions.create(
            messages = [{'role': 'system', 'content' : """We have received a video of a person making a speech."""+
                                                """We have extracted metrics that make up the attributes, dimensions and characteristics that are meaningful for evaluating """+objetivo+
                                                """of the person in the video who must talk about """+tema+
                                                """If the topic is not in English, you must translate it to English. The video has been evaluated in the following aspects:"""+
                                                """. Your task is, from the information we provide, to generate a report in the following order:"""+
                                                """1) Detailed explanation of all aspects evaluated in the video. You have to indicate why each aspect has obtained its results in the evaluation."""+
                                                """2) Result of the total evaluation with regard to """+objetivo+
                                                """The report can be as long as you consider, but keep in mind that the text that the user will provide you with is known only to the user. """+
                                                """You must generate a report that is understandable to anyone. The report should be a continuous text and in English."""},
                        {'role': 'user', 'content' : prompt}],
        
            model = 'gpt-3.5-turbo'
        )
    
        
    chat_response = completion.choices[0].message.content
    return chat_response


