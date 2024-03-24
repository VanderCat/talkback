from elevenlabs_unleashed.manager import ELUAccountManager
from elevenlabs import generate, set_api_key, play, api

eluac = ELUAccountManager(set_api_key, nb_accounts= 2) # Creates a queue of API keys
eluac.next() # First call will block the thread until keys are generated, and call set_api_key

def narrate(voice, text, model="eleven_multilingual_v2"):
    try:
        audio = generate(
            text=text,
            voice=voice,
            model=model
        )
    except api.error.RateLimitError as e:
        print("[ElevenLabs] Maximum number of requests reached. Getting a new API key...")
        eluac.next() # Uses next API key in queue, should be instant as nb_accounts > 1, and will generate a new key in a background thread.
        narrate(voice, text, model)
        return

    print("[ElevenLabs] Starting the stream...")
    with open("response.mp3", mode="wb") as f:
        f.write(audio)