# ECE884 Deep Learning for Denoising of Audio Files
# Introduction:
This project focuses on the automatic function of denoising audio signals using machine learning, and feeding clean audio output to a speech to text recognition model to calculate word error rate, compared to noisy audio.
This project features the CNN created by Daitan Innovations (Link here), with the evaluation metrics performed using Meta's Wav2Vec. 
# Pre-Requisites:
links to the databases utilized:

Mozilla Common Voice: https://commonvoice.mozilla.org/en/datasets

UrbanSound8k: https://urbansounddataset.weebly.com/urbansound8k.html

Noizeus: https://ecs.utdallas.edu/loizou/speech/noizeus/

Note: We recommoned using one of the smaller datasets from Mozilla Common Voice as the larger datasets may take an inordinate amount of time to download and uzip, potentially weeks. Though, naturally, the larger the dataset, the better the resulting model

Note: We recommend using the cpu version of tensorflow as it is easier get running, though your model will train slower depending on your desktop "pip install tensorflow-cpu"

# Methodolgy:

The logos behind our approach is to train a Network to learn the mapping between a given noisy audio file and it's clean counterpart. The noisy files are generated by superimposing differnt types of noise onto already clear recordings of speech
![image](https://user-images.githubusercontent.com/101994992/166852539-5c1daf92-b389-4539-8be7-4c240ac01b72.png)


# Architecture:
The model is based on an encoder-decoder architecture, both containing repeated blocks of Convolution, ReLU, and Batch Normalization. In total, the network contains 16 of such blocks — which adds up to 33K parameters. 
The model is fully convolutional, with no pooling or upscaling layers. The encoder-decoder layers are redundantly cascaded. The audio features extracted through convolution are expanded to a higher dimension with the encoder layers, then compressed down back to the original dimensionality of the noisy input.
The model partitions the noisy audio into overlapping windows of frequency domain data as the input batch. This allows the model to autoregressively learn the output based on past information and is generally more powerfully then simply feeding the entire audio file.

There are skip connections between some of the encoder and decoder blocks. These skip connections speed up convergence by letting the network learn the residuals between the layers, and also reduce vanishing gradients 

With a given input of shape of 129 x 8, convolution is only performed along the frequency axis. This ensures that the frequency axis remains constant during forward propagation

We then optimize the mean squared error (MSE) between the output of the network and the target (clean audio) signals using Adam Optimizer. Learning Rate: 3e-4, Stride Length: 1x1

# How to Run the Code:
Run These Scripts in this order; create_dataset, Train_Model, GenerateDenoisedAudio. The first two can be ran in whichever IDE you favor, but the last file must be ran in Juypter Lab or Juypter Notebook.

Note1: The Pathways that these scripts use to access our data must be changed to reflect the install pathways on your machine


Note2: To utilize the evaluation portion, it is required that you install the transformers, numpy, librosa, tensorflow, tokenizer, and torchaudio (pip install). Next import timeit, librosa, os, re, and torch.

Note 3:From Transformwers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer (imports wav2vec)
Note 4: To obtain word error rates for evaluation pip install jiwer and from jiwer import wer

Note 5: Import xlswriter as xlswr (creates and excel file to obtain word error rate results)

Note 6: If other audio files are being utilized, the reference text must be updated (line 106 to 111). Folder names, file locations for audio, must be updated as well.





# Evaluation:
To evaluate whether our denoised file performed better than the noisy file. Wav2Vec was utilized as an evaluation script. Wav2Vec was provided a clean audio file to obtain to obtain the error rate. Next the noisy audio file was processed through Wav2Vec to obtain a basis of perfomance reduction. Lastly the denoised audio file was processed through Wav2Vec to obtain the final results for improvement. To identify the improvement between the noisy and denoised files we apply the formula for Word Error Rate. This is an industry standard for calcuating how accurate a speech can be interpreted. 

![image](https://user-images.githubusercontent.com/101994705/166613580-ffccccb4-6b35-41b4-ae74-74e1a8a0e84c.png)

The evaluation script evaluates the following noise samples: airport, babble, car, exhibition, restaurant, street, train, and train station.

The sentences utilized for evaluation are the following: 
1. HE KNEW THE SKILL OF THE GREAT YOUNG ACTRESS
2. WE FIND JOY IN THE SIMPLEST THINGS
3. THE DRIP OF THE RAIN MADE A PLEASANT SOUND
4. THE FRIENDLY GANG LEFT THE DRUG STORE
5. CLAMS ARE SMALL ROUND SOFT AND TASTY
6. BRING YOUR BEST COMPASS TO THE THIRD CLASS

In the figures below are the results from utilizing the denoised neural network vs the noisy file. 

In the graph below, we see that the denoised data file improved a majority of the sentences Word Error Rates when compared to Wav2Vec just interpreting the noisy files.
![image](https://user-images.githubusercontent.com/101994705/166851695-13e3c0a1-727c-4b75-bd3d-2d2c0b130bc0.png)

The sentences obtained from denoised files and noisy files for the train station are displayed below with the error rate associated to each sentence: 

**SP02:** 

Ground Truth: HE KNEW THE SKILL OF THE GREAT YOUNG ACTRESS

Noisy Interpretation: HHE KNEW THE PR OF THE GREAT YOUNG MAN  - Error Rate: 33.3%

Denoised Interpretation: HE KNEW THE TILL OF THE GREAT YOUNG ASTRI  - Error Rate: 22.2%<br/>


SP07:
Ground Truth: WE FIND JOY IN THE SIMPLEST THINGS

Noisy Interpretation: WO I BOY  B  - Error Rate: 100%

Denoised Interpretation: WE FINDS YA WAY IN THE EBLET THEN  - Error Rate: 71%<br/>


SP12:
Ground Truth: THE DRIP OF THE RAIN MADE A PLEASANT SOUND

Noisy Interpretation: THE DRIP OF THE RAIN MADING LIV WIT DOWN  - Error Rate: 44.4%

Denoised Interpretation: THE DRIP OF THE RAIN NATING LET ENT DOWN  - Error Rate: 44.4%<br/>


SP18:
Ground Truth: THE FRIENDLY GANG LEFT THE DRUG STORE

Noisy Interpretation: EFRIENDLY GAME O TAT THE DUG  - Error Rate: 85.714%

Denoised Interpretation: THE FREELY GANG TAT THE DRAGONTO - Error Rate: 57.14%<br/>


SP21:

Ground Truth: CLAMS ARE SMALL ROUND SOFT AND TASTY

Noisy Interpretation: AAN BER EM ALL THE MAN AT AGOT O HIN  - Error Rate: 142.85%

Denoised Interpretation: LANBER AN USUAL MANIT ACOD AQE   - Error Rate: 100%<br/>


SP27:

Ground Truth: BRING YOUR BEST COMPASS TO THE THIRD CLASS

Noisy Interpretation: BRING YOUR BECD CUPE TTO THE FAR CLAD - Error Rate: 62.5%

Denoised Interpretation: BRING YOUR BED CUMBERT TO THE TIRY CLAM  - Error Rate: 50.0%<br/>

As we can see 5 sentences showed improvement and only one sentence had zero improvement. From this data we were able to obtain an average of 21% improvement using the denoising neural network. 


In other noise settings we saw noticeable improvements. For example, the exhibition hall and the car noise dataset showed noticeable improvement when denoised and still allowed Wav2Vec to interpret the wav form information. From the exhibition hall graph, 66% of the sentences were improved compared to the noisy data and 33% of them had the same error rate. 

![image](https://user-images.githubusercontent.com/101994705/166609671-905ed608-bb5e-4362-8c76-87c3a5e5e165.png)


![image](https://user-images.githubusercontent.com/101994705/166609700-22cecae2-4cd9-4ba6-aaa7-ef21ff2c3c7e.png)

The overall results of the noisy data and denoised data are shown below: 

![image](https://user-images.githubusercontent.com/101994705/166611644-193f9d31-ff32-4de1-87a9-706a7e84f0d7.png)

![image](https://user-images.githubusercontent.com/101994705/166611647-400faf09-4439-4246-a45d-631ec367f51f.png)

From the data above our denoised files, shows improvement compared to the base noisy file data. 81% of the data was improved or remained the same with only 19 percented having higher word error rates in comparison to the the noisy files. To further improve and increase word error rate additional investigations should be held to increase clarity of the original sentence or prevent distortion. 

Citations: 
1. Subramanian, Dhilip. “Speech to Text with wav2vec 2.0.” KDnuggets, 2 Mar. 2021, https://www.kdnuggets.com/2021/03/speech-text-wav2vec.html. 

2. Loizou, Philipos C., et al. “Noizeus: A Noisy Speech Corpus for Evaluation of Speech Enhancement Algorithms.” Dallas, https://ecs.utdallas.edu/loizou/speech/noizeus/. 





