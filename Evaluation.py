# Import necessary library

# For managing audio file

#from lib2to3.pytree import _Results
import timeit
import librosa
import os
import re
import xlsxwriter as xlswr
from jiwer import wer
workboook = xlswr.Workbook(r"C:\\Users\\bmjet\\OneDrive\\Desktop\\Final Test Selection - ECE-884\\Results.xlsx")
sheet = workboook.add_worksheet()

#Importing Pytorch
import torch

#Importing Wav2Vec
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

def PathFinder():
    FileTypes = ['wav']
    SearchStrings = ['noisy']
    airport = []
    babble = []
    car = []
    exhibition = []
    restaurant = []
    street = []
    train = []
    station = []
    top = os.getcwd()
    print(top)
    filecount = 0

    for root, dirs, files in os.walk(top, topdown=False):
        for fl in files:
            currentFile = os.path.join(root, fl)
            for FileType in FileTypes:
                status = str.endswith(currentFile, FileType)
                if str(status) == 'True':
                    for SearchString in SearchStrings:
                        if str(SearchString in currentFile) == 'True':
                            #if str(currentFile) not in airport not in babble not in car not in exhibition not in restaurant not in street not in train not in station:
                            if str(currentFile) not in airport: 
                                if re.search("airport", currentFile):
                                    filecount = filecount + 1
                                    airport.append(currentFile)
                            if str(currentFile) not in babble:
                                if re.search("babble", currentFile):
                                    filecount = filecount + 1
                                    babble.append(currentFile)
                            if str(currentFile) not in car:
                                if re.search("car", currentFile):
                                    filecount = filecount + 1
                                    car.append(currentFile)
                            if str(currentFile) not in exhibition:
                                if re.search("exhibition", currentFile):
                                    filecount = filecount + 1
                                    exhibition.append(currentFile)
                            if str(currentFile) not in restaurant:
                                if re.search("restaurant", currentFile):
                                    filecount = filecount + 1
                                    restaurant.append(currentFile)
                            if str(currentFile) not in street:
                                if re.search("street", currentFile):
                                    filecount = filecount + 1
                                    street.append(currentFile)
                            if str(currentFile) not in train:
                                if re.search("train", currentFile):
                                    filecount = filecount + 1
                                    train.append(currentFile)
                            if str(currentFile) not in station:
                                if re.search("station", currentFile):
                                    filecount = filecount + 1
                                    station.append(currentFile)
    print("COUNT *************  ", filecount)
    return [airport, babble, car, exhibition, restaurant, street, train, station]

audiofiles = PathFinder()

# Loading the audio file
def Transcrib(audiopath):
    audio, rate = librosa.load(audiopath, sr = 16000)

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
    #print("\n")
    #print (transcription)
    return transcription

#WER Calculator
ref_txt= ["He knew the skill of the great young actress",
    "We find joy in the simplest things",
    "The drip of the rain made a pleasant sound",
    "The friendly gang left the drug store",
    "Clams are small round soft and tasty",
    "Bring your best compass to the third class"]
F_cnt = 0

#eval_text_file = open ("C:\\Users\\bmjet\\OneDrive\\Desktop\\Final Test Selection - ECE-884\\result_text", "wt")

cn = 0
cn2 = 1
for folder in audiofiles:
    
    for idx in range(len(folder)):
        #n = eval_text_file.write("Comparison for "+ audiofiles[F_cnt][idx] ) 
        if idx == 0:
            print("folder ", cn2)
            cn2 += 1
        print("noisy ", idx)    
        
                  
        ground_truth = ref_txt[idx]
        transcription = Transcrib(audiofiles[F_cnt][idx])

       # n = eval_text_file.write(ground_truth.upper())
        error = wer(ground_truth.upper(), transcription)


        #n = eval_text_file.write(str(error))
        path, name = os.path.split(audiofiles[F_cnt][idx])
        sheet.write(0, cn, "Comparison for \n"+ name )
        sheet.write(1, cn, ground_truth.upper())
        sheet.write(2, cn, str(error)  )
        cn += 1
    F_cnt += 1
workboook.close()

