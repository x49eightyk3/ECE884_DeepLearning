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
To evaluate whether our denoised file performed better than the noisy file. Wav2Vec was utilized as an evaluation script. Wav2Vec was provided a clean audio file to obtain to obtain the error rate. Next the noisy audio file was processed through Wav2Vec to obtain a basis of perfomance reduction. Lastly the denoised audio file was processed through Wav2Vec to obtain the final results for improvement. To identify the improvement between the noisy and denoised files we apply the formula for Word Error Rate. This is an industry standard for calcuating how accurate a speech can be interpreted. 

![image](https://user-images.githubusercontent.com/101994705/166613580-ffccccb4-6b35-41b4-ae74-74e1a8a0e84c.png)

The evaluation script evaluates the following noise samples: airport, babble, car, exhibition, restaurant, street, train, and train station.


In the figures below are the results from utilizing the denoised neural network vs the noisy file. 

In the graph below, we are able to see the performance of 6 sentences comparing the performance of the denoised file vs the noisy file interpreted through Wav2Vec. 
The denoised files performed worse in SP02 and SP07. This is due to Wav2Vec interpeting a majority of the wav forms with an WER of .1111. The denoised file reduced a majority of the noisy feedback but also reduced the sound and feedback that is used to interpret speech units. This results in Wav2Vec being unable to accurately interpret the wav forms. 


As we can see below, Wav2Vec in this case was able to pick up a majority of the noisy file but when denoising was applied Wav2Vec was not able to accurately interpret the wav files. 

SP01 - Ground Truth Statement: HE KNEW THE SKILL OF THE GREAT YOUNG ACTRESS

sp02_airport_sn5.wav - Noisy File - HE KNEW THE SKILL OF THE GREAT YOUNG MAN

Noisy0.wav - Denoised File - HO KNEW THE THILL OF HIS READ YOUNG ACT


Graph 1.
![image](https://user-images.githubusercontent.com/101994705/166605282-3c81673a-243c-4b6a-8c99-3939159a5726.png)


The figure below shows denoising applied to the last noise file "Bring your best compass to the third class". In this figure we are able to noticeably see that the additional feedback was reduced. Resulting in higher accuracy compared to the Wav2Vec interpreting the noisy file. 

![airport](https://user-images.githubusercontent.com/101994705/166605330-7dcf9cbd-f5f6-45e4-9295-6b7a621f29f5.PNG)


In other noise settings we saw noticeable improvements. For example, the exhibition hall and the car noise dataset showed noticeable improvement when denoised and still allowed Wav2Vec to interpret the wav form information. 

![image](https://user-images.githubusercontent.com/101994705/166609671-905ed608-bb5e-4362-8c76-87c3a5e5e165.png)


![image](https://user-images.githubusercontent.com/101994705/166609700-22cecae2-4cd9-4ba6-aaa7-ef21ff2c3c7e.png)

The overall results of the noisy data and denoised data are shown below: 

![image](https://user-images.githubusercontent.com/101994705/166611644-193f9d31-ff32-4de1-87a9-706a7e84f0d7.png)

![image](https://user-images.githubusercontent.com/101994705/166611647-400faf09-4439-4246-a45d-631ec367f51f.png)

From the data above our denoised files, shows improvement compared to the base noisy file data. 

Citations: 
1. Subramanian, Dhilip. “Speech to Text with wav2vec 2.0.” KDnuggets, 2 Mar. 2021, https://www.kdnuggets.com/2021/03/speech-text-wav2vec.html. 

2. Loizou, Philipos C., et al. “Noizeus: A Noisy Speech Corpus for Evaluation of Speech Enhancement Algorithms.” Dallas, https://ecs.utdallas.edu/loizou/speech/noizeus/. 





