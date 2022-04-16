from jiwer import wer

ground_truth = "hello world I am paul"
hypothesis = "hello duck by ham paul"

error = wer(ground_truth, hypothesis)

print (error)