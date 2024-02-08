from openai import OpenAI
import os
from dotenv import load_dotenv
import base64
import streamlit as st
import elevenlabs

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

def get_answer(messages):
    system_message = [{"role": "system", "content": "Give greetings to user then ask how are you and then You have to ask the user about their class and then show subjects according to there class then ask which subject they want to learn  then give the syllabus of that subject and ask him what he want he to learn from it and show him well detailed answer about his topic then give himg questions about his topic"}]
    messages = system_message + messages
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=messages
    )
    return response.choices[0].message.content

def speech_to_text(audio_data):
    with open(audio_data, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file
        )
    return transcript

elevenlabs_api_key = 'a2a3e5ff57a8af84d1ca905143618598'
elevenlabs.set_api_key(elevenlabs_api_key)

def text_to_speech(input_text, voice="Veda", settings=None):
    # Use Eleven Labs API for text-to-speech
    if settings:
        audio_data = elevenlabs.generate(input_text, voice=voice, settings=settings)
    else:
        audio_data = elevenlabs.generate(input_text, voice=voice)
    
    # Save the audio file
    webm_file_path = "temp_audio_play.mp3"
    with open(webm_file_path, "wb") as f:
        f.write(audio_data)
    
    return webm_file_path
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    md = f"""
    <audio autoplay>
    <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """
    st.markdown(md, unsafe_allow_html=True)