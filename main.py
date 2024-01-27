import config
import pyaudio
import soundfile as sf
import numpy as np
import whisper
import scipy
import g4f
 
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

whisper_model = whisper.load_model(config.WHISPER_MODEL)

buffer = b''
audStarted = False

counter = 0
def record():
    global counter
    global buffer
    global audStarted
    if(counter > 30):
        buffer = b''
        audStarted = False
        return True
    data = stream.read(CHUNK_SIZE)

    #print ("recording stopped")
    
    rawAudio = np.frombuffer(data, np.int16).astype(np.float32)

    audio = rawAudio*(1/32768.0)

    fileMean = 20*np.log10(np.sqrt(np.mean(np.absolute(rawAudio)**2)))
    #print(fileMean)

    if (fileMean > config.NOISE_LEVEL):
        #print(audio)
        buffer+=data
        audStarted = True
    elif(audStarted):
        audStarted = False
        return True
    counter+=1
    return False

print("listening...")
while True:
    recording = record()
    if (recording):
        print("running!")
        rawAudio = np.frombuffer(buffer, np.int16).astype(np.float32)
        audio = rawAudio*(1/32768.0)
        audio1 = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio1).to(whisper_model.device)
        _, probs = whisper_model.detect_language(mel)
        print(f"Detected language: {max(probs, key=probs.get)}")
        options = whisper.DecodingOptions(fp16=False, language="ru")
        result = whisper.decode(whisper_model, mel, options)
        scipy.io.wavfile.write("test.wav", 16000, rawAudio.astype('int16'))
        # print the recognized text
        buffer = b''
        print(result.text)
        response = g4f.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT}, {"role": "user", "content": result.text}],
        )
        print(response)

stream.stop_stream()
stream.close()
audio.terminate()