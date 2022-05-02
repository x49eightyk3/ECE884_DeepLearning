# ECE884 Deep Learning for Denoising of Audio Files
# Introduction:
This project focuses on the automatic function of denoising audio signals using machine learning, and feeding clean audio output to a speech to text recognition model to calculate word error rate, compared to noisy audio.
This project features the CNN created by Daitan Innovations (Link here), with the evaluation metrics performed using Meta's Wav2Vec. 
# How to Run the Code:



To utilize the evaluation portion of the it is required to install the transformers library (pip install transformers). 

# Evaluation:
To evaluate whether our denoised file performed better than the noisy file. Wav2Vec was utilized as an evaluation script. Wav2Vec was provided a clean audio file to obtain to obtain the error rate. Next the noisy audio file was processed through Wav2Vec to obtain a basis of perfomance reduction. Lastly the denoised audio file was processed through Wav2Vec to obtain the final results for improvement. 







