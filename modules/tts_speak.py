from gtts import gTTS
from playsound import playsound
import uuid
import os

def tts_speak(text):
    print(f"💬 Speaking: {text}")
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang='zh-tw')
    tts.save(filename)
    playsound(filename)
    os.remove(filename)
