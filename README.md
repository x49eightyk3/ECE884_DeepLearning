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


In the figures below are the results from utilizing the denoised neural network vs the noisy file. 

![image](https://user-images.githubusercontent.com/101994705/166176172-722807a4-76d6-4a3a-955b-d64afbbc856a.png)

![airport](https://user-images.githubusercontent.com/101994705/166176352-d8f8a4bd-f7f4-4d19-b846-8c99eca8e15a.PNG)




![image](https://user-images.githubusercontent.com/101994705/166176195-32bd0d48-ccae-407c-a14d-6e1d2423853a.png)

![image](https://user-images.githubusercontent.com/101994705/166176203-33ea5d2b-0d31-495d-8946-7bb11eb3cef6.png)

![image](https://user-images.githubusercontent.com/101994705/166176212-148658db-fe9b-453d-9a26-fba5917cb0b8.png)

![image](https://user-images.githubusercontent.com/101994705/166176228-e55a0fa9-d0c2-4161-9d67-d43fc3cad316.png)

![image](https://user-images.githubusercontent.com/101994705/166176233-b30c88d2-4da6-4df0-84f5-b0a0221dd388.png)

![image](https://user-images.githubusercontent.com/101994705/166176240-13b99f8d-6938-4e62-842f-143bc49531ec.png)

![image](https://user-images.githubusercontent.com/101994705/166176243-48d2a18a-6f77-442b-a4c8-89f649b34279.png)








![babble](https://user-images.githubusercontent.com/101994705/166176364-4fb2a1c1-9cd6-4388-8f7a-29b752c572a5.PNG)

![car](https://user-images.githubusercontent.com/101994705/166176376-500a70d8-c799-4834-b774-5c3bd88898b6.PNG)

![ExhibitionHall](https://user-images.githubusercontent.com/101994705/166176380-54b4b885-6b28-468f-b26b-cf8d82dc1fb8.PNG)

![restuarant](https://user-images.githubusercontent.com/101994705/166176386-89497163-a08e-40bc-ae0d-a57d4abb07f3.PNG)

![street](https://user-images.githubusercontent.com/101994705/166176412-4f4618b5-cb19-4dd3-9fa1-5270203b7f48.PNG)

![street](https://user-images.githubusercontent.com/101994705/166176418-94ed45fd-ad2c-4ead-b8e9-fdbe109a43d1.PNG)

![street](https://user-images.githubusercontent.com/101994705/166176422-03cdc760-c009-4584-a7c9-a8f51db66473.PNG)




Citations: 
1. Subramanian, Dhilip. “Speech to Text with wav2vec 2.0.” KDnuggets, 2 Mar. 2021, https://www.kdnuggets.com/2021/03/speech-text-wav2vec.html. 

2. Loizou, Philipos C., et al. “Noizeus: A Noisy Speech Corpus for Evaluation of Speech Enhancement Algorithms.” Dallas, https://ecs.utdallas.edu/loizou/speech/noizeus/. 





