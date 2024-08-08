import streamlit as st
import numpy as np
import time
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch

# initialise device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# iunction to initialise ASR model
def init_asr_pipeline(model_name):
    return pipeline("automatic-speech-recognition", model=model_name, device=device)

# TTS model
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

model.to(device)
vocoder.to(device)

# embeddings
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# translate and synthesise
def translate(audio, asr_pipeline):
    outputs = asr_pipeline(audio, max_new_tokens=256, generate_kwargs={"task": "translate"})
    return outputs["text"]

def synthesise(text):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder
    )
    return speech.cpu().numpy()

def speech_to_speech_translation(audio, asr_pipeline):
    tic = time.perf_counter()
    translated_text = translate(audio, asr_pipeline)
    synthesised_speech = synthesise(translated_text)
    toc = time.perf_counter()
    st.write(f"Running time for S2S: {toc - tic:0.4f} seconds")
    return 16000, synthesised_speech

# CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# front-end
st.markdown("<div class='title'>Benchmarking Models</div>", unsafe_allow_html=True)
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### Upload an audio file for translation", unsafe_allow_html=True)

# model selection
model_options = {
    "Whisper-base": "openai/whisper-base",
    "Whisper-small": "openai/whisper-small",
    "Whisper Distil Large v3": "distil-whisper/distil-large-v3"
}

selected_model = st.selectbox("Select ASR Model", list(model_options.keys()))
asr_pipeline = init_asr_pipeline(model_options[selected_model])

# file upload
audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
if st.button("Translate Uploaded Audio"):
    if audio_file:
        audio_bytes = audio_file.read()
        sampling_rate, translated_speech = speech_to_speech_translation(audio_bytes, asr_pipeline)
        st.audio(translated_speech, format="audio/wav", sample_rate=sampling_rate)
    else:
        st.warning("Please upload an audio file first.")
st.markdown("</div>", unsafe_allow_html=True)
