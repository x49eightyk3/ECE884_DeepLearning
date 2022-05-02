# ECE884 Deep Learning for Denoising of Audio Files
# Introduction:
This project focuses on the automatic function of denoising audio signals using machine learning, and feeding clean audio output to a speech to text recognition model to calculate word error rate, compared to noisy audio.
This project features the CNN created by Daitan Innovations (Link here), with the evaluation metrics performed using Meta's Wav2Vec. 
# Pre-Requisites:
Add pathways to all data used here

Note: We recommoned using one of the smaller datasets from Mozilla Common Voice as the larger datasets may take an inordinate amount of time to download and uzip, potentially weeks. Though, naturally, the larger the dataset, the better the resulting model
# How to Run the Code:
Run These Scripts in this order; create_dataset, Train_Model, GenerateDenoisedAudio. The first two can be ran in whichever IDE you favor, but the last file must be ran in Juypter Lab or Juypter Notebook.

Note1: The Pathways that these scripts use to access our data must be changed to reflect the install pathways on your machine


Note2: To utilize the evaluation portion, it is required that you install the transformers library (pip install transformers). Next import timeit, librosa, os, re, and torch

Note 3:From Transformwers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer (imports wav2vec)
Note 4: To obtain word error rates for evaluation pip install jiwer and from jiwer import wer

Note 5: Import xlswriter as xlswr (creates and excel file to obtain word error rate results)

Note 6: If other audio files are being utilized, the reference text must be updated (line 106 to 111). Folder names, file locations for audio, must be updated as well.





# Evaluation:
To evaluate whether our denoised file performed better than the noisy file. Wav2Vec was utilized as an evaluation script. Wav2Vec was provided a clean audio file to obtain to obtain the error rate. Next the noisy audio file was processed through Wav2Vec to obtain a basis of perfomance reduction. Lastly the denoised audio file was processed through Wav2Vec to obtain the final results for improvement. 

The evaluation script evaluates the following noise samples: airport, babble, car, exhibition, restaurant, street, train, and train station.

![image](https://user-images.githubusercontent.com/101994705/166176172-722807a4-76d6-4a3a-955b-d64afbbc856a.png)

![image](https://user-images.githubusercontent.com/101994705/166176195-32bd0d48-ccae-407c-a14d-6e1d2423853a.png)

![image](https://user-images.githubusercontent.com/101994705/166176203-33ea5d2b-0d31-495d-8946-7bb11eb3cef6.png)

![image](https://user-images.githubusercontent.com/101994705/166176212-148658db-fe9b-453d-9a26-fba5917cb0b8.png)

![image](https://user-images.githubusercontent.com/101994705/166176228-e55a0fa9-d0c2-4161-9d67-d43fc3cad316.png)

![image](https://user-images.githubusercontent.com/101994705/166176233-b30c88d2-4da6-4df0-84f5-b0a0221dd388.png)

![image](https://user-images.githubusercontent.com/101994705/166176240-13b99f8d-6938-4e62-842f-143bc49531ec.png)

![image](https://user-images.githubusercontent.com/101994705/166176243-48d2a18a-6f77-442b-a4c8-89f649b34279.png)







