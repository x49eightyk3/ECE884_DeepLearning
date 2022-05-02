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


Note2: To utilize the evaluation portion, it is required that you install the transformers library (pip install transformers). 

# Evaluation:
To evaluate whether our denoised file performed better than the noisy file. Wav2Vec was utilized as an evaluation script. Wav2Vec was provided a clean audio file to obtain to obtain the error rate. Next the noisy audio file was processed through Wav2Vec to obtain a basis of perfomance reduction. Lastly the denoised audio file was processed through Wav2Vec to obtain the final results for improvement. 







