import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.tts_speak import tts_speak
from modules.door_checker import check_door_status
import speech_recognition as sr

SAMPLE_RATE = 16000
RECORD_SECONDS = 6

def handle_command():
    print("🎤 準備錄音使用者指令，請在嗶聲後開始說話...")
    time.sleep(1.5)
    print("🔔 嗶！開始錄音中...")

    recognizer = sr.Recognizer()
    mic = sr.Microphone(sample_rate=SAMPLE_RATE)

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source, timeout=RECORD_SECONDS)
            print("✅ 錄音完成，開始辨識...")
    except Exception as e:
        print(f"❌ 錄音失敗: {e}")
        tts_speak("錄音時發生錯誤")
        return

    try:
        command_text = recognizer.recognize_google(audio, language='zh-TW').strip().lower()
        print(f"📄 Google 語音辨識結果: {command_text}")
    except sr.UnknownValueError:
        print("❌ 無法理解語音")
        tts_speak("不好意思，我聽不懂你說什麼。")
        return
    except sr.RequestError as e:
        print(f"❌ 語音辨識服務錯誤: {e}")
        tts_speak("語音辨識服務連線失敗")
        return

    if "前門" in command_text and "後門" in command_text:
        intent = "check_door_lock"
        door = "both"
    elif "前門" in command_text:
        intent = "check_door_lock"
        door = "front"
    elif "後門" in command_text:
        intent = "check_door_lock"
        door = "back"
    else:
        intent = "unknown"

    if intent == "check_door_lock":
        result_text = check_door_status()
        tts_speak(result_text)
    else:
        tts_speak("不好意思，我聽不懂你的指令。")
