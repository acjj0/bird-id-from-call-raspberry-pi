# Created minimal listener & classifier to run on edge (Raspberry Pi, Jetson Nano, etc) that may not run full tensorflow
# Requires pyaudio (first run "sudo apt-get install python3-pyaudio"), tflite_runtime, numpy, and psycopg2 only
# For psycopg2 on edge devices, you need to pip install psycopg2-binary
from dotenv import load_dotenv
import pyaudio
import numpy as np
from tflite_runtime.interpreter import Interpreter

import psycopg2 # Requires psycopg2-binary on edge devices like Raspberry Pi
from datetime import datetime, timezone
from os import environ

load_dotenv()
labels = open("./static/assets/mobilenet_labels.txt",'r').read().splitlines()
interpreter = Interpreter(model_path="./static/assets/mobilenet.tflite")

SCORE_THRESHOLD = environ.get('SCORE_THRESHOLD')
DEVICE_ID=environ.get('DEVICE_ID')
DATABASE_URL=environ.get('DATABASE_URL').replace('postgres://', 'postgresql://')

def listener():
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    SAMPLE_RATE = 16000
    CHUNK = SAMPLE_RATE // 2
    RECORD_SECONDS = 5
    NFRAMES = 6

    # initialize pyaudio
    pa = pyaudio.PyAudio()

    print('opening stream...')
    stream = pa.open(format = FORMAT,
                     channels = CHANNELS,
                     rate = SAMPLE_RATE,
                     input = True,
                     frames_per_buffer = CHUNK)

    try:
        while True:
            print("Listening...")

            frames = []

            # The 6 here is totally force-fitted to match input.
            for i in range(0, NFRAMES):
                data = stream.read(CHUNK)
                frames.append(data)

            # np.fromstring DeprecationWarning: The binary mode of fromstring is deprecated,
            # as it behaves surprisingly on unicode inputs. Use frombuffer instead
            # Hence,
            buffer = b''.join(frames)
            audio_data = np.frombuffer(buffer, dtype=np.float32)

            # Cannot set tensor: Dimension mismatch. Got 1 but expected 2 for input 0
            # Hence, add a new dimension with this line
            audio_data = np.expand_dims(audio_data, 0)

            # run inference on audio data
            classifier(audio_data)

    except KeyboardInterrupt:
        print("exiting listener...")

    stream.stop_stream()
    stream.close()
    pa.terminate()


def classifier(audio_data):
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], np.array(audio_data, dtype='float32'))

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    results = dict(zip(labels, output_data[0]))
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    if sorted_results[0][0] != "nocall" and sorted_results[0][1] > SCORE_THRESHOLD:
        write_birdheard_to_postgres(sorted_results[0][0], sorted_results[0][1].item())

def write_birdheard_to_postgres(ebird_code, confidence):
    print(f"Writing {ebird_code} with confidence score {confidence} to database ...")

    db_connection = psycopg2.connect(DATABASE_URL)
    cur = db_connection.cursor()

    cur.execute('''INSERT INTO birds_heard (ebird_code, confidence, when_heard, device_id) VALUES (%s,%s,%s,%s)''',
        (ebird_code, confidence, datetime.now(timezone.utc), DEVICE_ID))
    db_connection.commit()
    cur.close()
    db_connection.close()
def main():
    listener()

if __name__ == '__main__':
    main()
