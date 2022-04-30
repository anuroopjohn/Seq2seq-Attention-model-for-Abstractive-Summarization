# Seq2seq-Attention-model-for-Abstractive-Summarization

This repository consists of the implementation of a sequence to sequence with attention model using PyTorch for an abstractive summarization system on CNN/Daily Mail Dataset, which consists of more than 300K news articles and each of them is paired with several highlights, known as multi-sentence summaries.
Distribution of number of datasamples in the dataset are given below.

<img width="272" alt="image" src="https://user-images.githubusercontent.com/30875547/166088256-e55f7cec-8033-4fff-ad05-51194a0257c3.png">

The Original data ```Seq2seq-with-Attention-model-for-Abstractive-Summarization/data``` conatins the source, target data for the train, validation, test splits.

## Training the model
After cloning the github repo, kingly navigate to ```Seq2seq-with-Attention-model-for-Abstractive-Summarization/code``` and run the the following command to train the model.

```python main.py```

The model was trained using a Nvidia-3090 gpu with 24 Gb memory. Both the memory, CPU ultilization was above 90% during the training, hence inorder to train the model in lower gpu configuration, kindly reduce the ```MAX_VOCAB, BATCH_SIZE, ENC_EMB_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM```, which can found inside the ```python main.py``` script.

## Evaluation
Kindly run the below script to evaluate the model's prediction in ROUGE metric.

```python evaluate.py```


