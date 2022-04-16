# Import necessary library

# For managing audio file
import timeit
import librosa

from jiwer import wer


#Importing Pytorch
import torch

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# Loading the audio file

audio, rate = librosa.load("C:\\Users\\bmjet\\OneDrive\\Desktop\\ECE-884\\Example_File_Noisy_Keyboard_Click.wav", sr = 16000)


# Importing Wav2Vec pretrained model

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# Taking an input value

input_values = tokenizer(audio, return_tensors = "pt").input_values


# Storing logits (non-normalized prediction values)
logits = model(input_values).logits

# Storing predicted ids
prediction = torch.argmax(logits, dim = -1)


# Passing the prediction to the tokenzer decode to get the transcription
transcription = tokenizer.batch_decode(prediction)[0]
print("\n")
print (transcription)


#WER Calculator
ground_truth = "He wrote down a long list of items the drip of rain made a pleasant sound smoke poured out of every crack hats are worn to tea and not to dinner the clothes dried on a thin wooden rack"
hypothesis = transcription


print (ground_truth.upper())

error = wer(ground_truth.upper(), hypothesis)

print (error)

