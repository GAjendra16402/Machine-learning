import streamlit as st
import os
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Float feature initialization
float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello Student!, How are you? Please tell me your name"}
        ]
        st.session_state.first_query = True

def generate_test_questions():
    # Modify this function to generate your test questions dynamically based on the completed topic
    questions = [
        "Question 1: True or False - [Statement]",
        "Question 2: True or False - [Statement]",
        "Question 3: True or False - [Statement]",
        "Question 4: True or False - [Statement]",
        "Question 5: True or False - [Statement]"
    ]
    return questions

initialize_session_state()

st.title("Student Assistant Bot 🧠💡")

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)

        transcript = speech_to_text(webm_file_path)
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking🤔..."):
            final_response = get_answer(st.session_state.messages)
        
        with st.spinner("Generating audio response..."):    
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)

        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        os.remove(audio_file)

        # Check if the topic is completed and generate test questions
        if "completed_topic" in st.session_state:
            test_questions = generate_test_questions()
            st.write("Here are your test questions:")
            for question in test_questions:
                st.write(question)
