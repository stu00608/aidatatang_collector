"""
A command line interface to record audio for each transcript in material transcript 
and save it as aidatatang_200zh format for training MockingBird model.

shen_{number} 以后你是男孩子
"""

import os
import sys
import tty
import wave
import random
import argparse
import pyaudio
import noisereduce as nr
import speech_recognition as sr
from pydub import AudioSegment
from scipy.io import wavfile

alphabets = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Set up the microphone
r = sr.Recognizer()
mic = sr.Microphone()


def trim_audio(input_file_path, output_file_path):

    sound = AudioSegment.from_file(
        input_file_path, format="wav")

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())

    duration = len(sound)
    print(
        f"start_trim: {start_trim}, end_trim: {end_trim}, duration: {duration}")
    trimmed_sound = sound[start_trim:duration-end_trim]
    trimmed_sound.export(output_file_path, format="wav")


def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0  # ms

    assert chunk_size > 0  # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size

    return trim_ms


def write_wav(fs, data, filename):
    with wave.open(filename, 'wb') as wav_file:
        # Set the audio parameters
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(fs)

        # Write the audio data
        wav_file.writeframes(data)


def record_noise(filename):
    # Set the parameters for the audio recording
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 3

    print("Recording noise...")

    # Initialize the PyAudio object
    p = pyaudio.PyAudio()

    # Open a streaming stream to get audio data from the microphone
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Create a list to store the audio data
    data = []

    # Start recording
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        audio_data = stream.read(CHUNK)
        data.append(audio_data)

    # Stop the stream and close it
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

    # Save the recorded audio data to a WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(data))
    wf.close()

    print("Done")


def record_audio(raw_output_path, wav_output_path, noise_file_path=None):
    # Record audio
    try:
        with mic as source:
            r.adjust_for_ambient_noise(source)
            print("Say something!")
            audio = r.listen(source)
            print("Got it! Now saving...")
    except Exception as e:
        print("Failed to record audio: ", e)
        return False
    # Write audio to a WAV file
    try:
        with open(raw_output_path, "wb") as f:
            f.write(audio.get_wav_data())
    except Exception as e:
        print("Failed to save audio: ", e)
        return False

    if noise_file_path == None:
        return

    input_rate, input_data = wavfile.read(raw_output_path)
    noise_rate, noise_data = wavfile.read(noise_file_path)
    data = nr.reduce_noise(y=input_data, sr=input_rate,
                           y_noise=noise_data, prop_decrease=0.99, n_jobs=2)
    tmp_denoised_audio = "/tmp/denoised_audio.wav"
    write_wav(input_rate, data, tmp_denoised_audio)
    trim_audio(tmp_denoised_audio, wav_output_path)

    return True


def get_random_string():
    return "".join(random.sample(alphabets, 10))


def write_to_file(file_name, string_list):
    with open(file_name, "w", encoding="utf-8") as f:
        for s in string_list:
            f.write(s + "\n")


def create_empty_file(file_name):
    if not os.path.exists(file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            f.write("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", type=str,
                        help="Name of the speaker.", required=True)
    parser.add_argument("--transcript", "-t", type=str,
                        default="contents.txt", help="Transcript content split in line.")
    args = parser.parse_args()

    if os.path.exists("noise.wav"):
        os.system("clear")
        print("Do you want to record noise sample audio? This will replace the noise.wav in this folder. (y/n)")
        tty.setcbreak(sys.stdin)
        key = ord(sys.stdin.read(1))  # key captures the key-code
        if key == ord('y'):
            record_noise("noise.wav")
    else:
        record_noise("noise.wav")

    dataset_root_path = os.path.join(f"{args.name}_dataset", "aidatatang_200zh")

    # Path for saving microphone record audio files.
    audio_folder = os.path.join(dataset_root_path, "corpus", "train", "wav")
    raw_audio_folder = os.path.join(dataset_root_path, "corpus", "raw", "wav")
    # Path for saving transcript files.
    transcirpt_folder = os.path.join(dataset_root_path, "transcript")

    # Create output folder if not exist.
    os.makedirs(dataset_root_path, exist_ok=True)
    os.makedirs(transcirpt_folder, exist_ok=True)
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(raw_audio_folder, exist_ok=True)

    # Transcript and passed transcript file path.
    transcript_path = os.path.join(
        transcirpt_folder, "aidatatang_200_zh_transcript.txt")
    passed_transcript_path = os.path.join(
        transcirpt_folder, "passed_transcript.txt")

    # Read contents
    with open(args.transcript, "r", encoding="utf-8") as f:
        contents = f.read().splitlines()
        contents.sort()
        for i, c in enumerate(contents):
            if not (c[0] in alphabets):
                print(i)
                contents = contents[i:]
                break
        random.shuffle(contents)

    # Create transcript file if not exist.
    create_empty_file(transcript_path)
    create_empty_file(passed_transcript_path)

    # Match current transcript and corpus,
    # if there is any file not exist in corpus,
    # remove it from transcript. This means if user want to delete a record,
    # just need to delete the wav file.
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().splitlines()
        transcirpt = [t for t in transcript if os.path.exists(
            os.path.join(audio_folder, t.split(" ")[0] + ".wav"))]
        print(f"Transcript found: {transcirpt}")
    write_to_file(transcript_path, transcirpt)

    # Reading transcript.
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().splitlines()

        transcript_contents = [" ".join(l.split(" ")[1:])
                               for l in transcript]
        if transcript and transcript[0].split(" ")[0].split("_")[0] != args.name:
            print("Transcript name not match current dataset.")
            sys.exit()

    with open(passed_transcript_path, "r", encoding="utf-8") as f:
        passed_transcript = f.read().splitlines()

    print("Transript: ", transcript)
    print("Transcript length: ", len(transcript))

    index = len(transcript)
    record_count = 0
    for c in contents:
        if c in transcript_contents or c in passed_transcript:
            # print(f"{c} Already exist.")
            continue

        os.system("clear")
        print(
            f"""\n\n{c}\n\nPress:\n[p]: Record and save\n[n]: Pass this text.\n[q]: Quit.\n\n""")

        while True:
            tty.setcbreak(sys.stdin)
            key = ord(sys.stdin.read(1))  # key captures the key-code
            # based on the input we do something - in this case print something
            if key == ord('p'):
                tag = f"{args.name}_{get_random_string()}"
                file_name = tag + ".wav"
                wav_file_path = os.path.join(audio_folder, file_name)
                raw_file_path = os.path.join(raw_audio_folder, file_name)

                # Record audio
                if record_audio(raw_file_path, wav_file_path, noise_file_path="noise.wav"):
                    print("Record saved.")
                else:
                    print("Record failed.")
                    continue

                index += 1

                transcript_data = f"{tag} {c}"

                transcript_contents.append(c)
                transcript.append(transcript_data)
                write_to_file(transcript_path, transcript)
                record_count += 1
                break
            if key == ord('n'):
                print("Go to next sentence.")
                passed_transcript.append(c)
                write_to_file(passed_transcript_path, passed_transcript)
                record_count += 1
                break
            if key == ord('q'):
                print("Quit.")
                sys.exit()

    if record_count == 0 or record_count == len(contents):
        print("Record completed!")
