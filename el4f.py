import requests
import pydub
from pydub.playback import play

BASE_URL = "https://api.elevenlabs.io/v1/"

def narrate(voiceid, text, model="eleven_multilingual_v2"):
    resp = requests.post(BASE_URL+"text-to-speech/"+voiceid, json={
        "text": text,
        "model_id": model
    })
    with open("response.mp3", mode="wb") as f:
        f.write(resp.content)