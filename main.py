import torch
import numpy as np
import sounddevice as sd
import wave
from speechbrain.inference import SpeakerRecognition
import os

# Set the local strategy to copy to avoid symlink issues
os.environ["SPEECHBRAIN_LOCALSTRATEGY"] = "COPY"

# Load the Speaker Recognition model with "copy" strategy to avoid symlink issues
model = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb", 
    savedir="tmp_model", 
    run_opts={"local_strategy": "copy"}  # Forces copying instead of symlink
)

# Function to record audio from the microphone
def record_audio(filename, duration=7, samplerate=16000):
    print(f"Recording for {duration} seconds...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    
    wavefile = wave.open(filename, 'wb')
    wavefile.setnchannels(1)    
    wavefile.setsampwidth(2)  # 16-bit audio
    wavefile.setframerate(samplerate)
    wavefile.writeframes(audio.tobytes())
    wavefile.close()

# Dictionary to store the speaker's name and their recorded audio file
known_speakers = {}

def enroll_speakers(num_speakers):
    for _ in range(num_speakers):
        name = input("Enter your name: ")
        filename = f"{name.lower()}.wav"
        print(f"Recording voice sample for {name}...")
        record_audio(filename, duration=7)  # Record voice for enrollment (7 seconds)
        known_speakers[name] = filename
        print(f"✅ Enrollment complete for {name}.")

# Recognize the speaker from test audio
def recognize_speaker(test_audio):
    best_match = None
    best_score = float('-inf')

    # Verify the speakers by providing the file paths, not tensors
    for name, ref_audio in known_speakers.items():
        score, _ = model.verify_files(ref_audio, test_audio)
        print(f"Comparing with {name}: Score = {score.item()}")  # For debugging
        if score > best_score:
            best_score = score
            best_match = name

    # Increased the threshold for a more strict match
    if best_score > 0.95:  # This makes sure the match is very high
        return best_match
    else:
        return "Unknown"  # Return "Unknown" if the speaker is not recognized with enough confidence

# Real-time speaker recognition loop
def live_speaker_recognition():
    print("\n🎙️ Listening for speaker...")
    record_audio("test.wav", duration=7)  # Record a test sample (7 seconds)
    speaker = recognize_speaker("test.wav")  # Identify the speaker
    print(f"🎤 Speaker Identified: {speaker}")

# Ask for the number of voices to record
num_speakers = int(input("How many voices do you want to record? "))

# Run speaker enrollment for each user
enroll_speakers(num_speakers)

# Continuously listen and identify speakers after enrollment
while True:
    live_speaker_recognition()
    print("\n⏹️ Speaker recognized, stopping...")
    break  # Stop the loop once a speaker is recognized or it says "Unknown"
