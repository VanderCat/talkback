import config
import pyaudio
import soundfile as sf
import numpy as np
import whisper
import g4f
import el4f
#import eleven2
import compat
import pydub
import pydub.playback as pyb
#from elevenlabs_unleashed.tts import UnleashedTTS

#tts = UnleashedTTS(nb_accounts=2, create_accounts_threads=2)

from so_vits_svc_fork.inference.core import Svc
from so_vits_svc_fork.utils import get_optimal_device

svc_device = get_optimal_device()
 
RATE = 16000
CHUNK_SIZE = 16000
FORMAT = pyaudio.paInt16
FORMATOUT = pyaudio.paInt16

device_index = 2
audio = pyaudio.PyAudio()

print("----------------------record device list---------------------")
info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

print("-------------------------------------------------------------")

index = int(input())
print("recording via index "+str(index))

stream =  audio.open(
        format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE, input_device_index=index
    )

print("loading whisper")
whisper_model = whisper.load_model(config.WHISPER_MODEL)

print("loading svc")
modelDefenition = compat.ModelDefenition(config.SVC_SETTINGS)
svc_model = Svc(
    net_g_path=modelDefenition.model.as_posix(),
    config_path=modelDefenition.config.as_posix(),
    cluster_model_path=modelDefenition.cluster.as_posix()
    if modelDefenition.cluster
    else None,
    device=svc_device
)

buffer = b''
audioBuff = None
audStarted = False

counter = 0
def record():
    global counter
    global audioBuff
    if(counter > 30):
        return True
    data = stream.read(CHUNK_SIZE)

    #print ("recording stopped")
    
    rawAudio = np.frombuffer(data, np.int16).astype(np.float32)

    fileMean = 20*np.log10(np.sqrt(np.mean(np.absolute(rawAudio)**2)))
    print(fileMean)

    if (fileMean > config.NOISE_LEVEL):
        if (audioBuff is None):
            audioBuff = rawAudio
        else:
            audioBuff = np.append(audioBuff, rawAudio)
        counter+=1
    elif not (audioBuff is None):
        return True
    return False

print("listening...")
while True:
    recording = record()
    if (recording):
        print("running!")
        audio = audioBuff*(1/32768.0)
        audioBuff = None
        audio1 = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio1).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions(fp16=False, language="ru")
        result = whisper.decode(whisper_model, mel, options)
        # print the recognized text
        buffer = b''
        print(result.text)
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": result.text}],
        )
        print(response)
        el4f.narrate(config.ELEVENLABS_VOICE_MODEL, response)
        #eleven2.narrate("John", response)
        
        audioCombined = sf.SoundFile("response.mp3").read()
        
        audioCombined = svc_model.infer_silence(
            audioCombined.astype(np.float32),
            speaker=modelDefenition.speaker,
            transpose=modelDefenition.transpose,
            auto_predict_f0=modelDefenition.auto_predict_f0,
            cluster_infer_ratio=modelDefenition.cluster_infer_ratio,
            noise_scale=modelDefenition.noise_scale,
            f0_method=modelDefenition.f0_method,
            db_thresh=modelDefenition.db_thresh,
            pad_seconds=modelDefenition.pad_seconds,
            chunk_seconds=modelDefenition.chunk_seconds,
            absolute_thresh=modelDefenition.absolute_thresh,
            max_chunk_seconds=modelDefenition.max_chunk_seconds,
        )
        sf.write("final.wav", audioCombined, svc_model.target_sample)

        seg = pydub.AudioSegment.from_wav("final.wav")
        pyb.play(seg)
        

stream.stop_stream()
stream.close()
audio.terminate()