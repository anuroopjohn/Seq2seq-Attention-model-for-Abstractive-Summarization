import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
from tqdm import tqdm
import pandas as pd
from read_data import read_data,preprocess
from create_vocab import create_vocab
from data_loader import data_loader
from model import load_model
from train import train, evaluate
from inference import translate_sentence, calculate_bleu



MAX_VOCAB = 15000
BATCH_SIZE=4
ENC_EMB_DIM = 64
DEC_EMB_DIM = 64
ENC_HID_DIM = 128
DEC_HID_DIM = 128
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_EPOCHS = 5
CLIP = 1
summarization_lenth = 50
model_path = '/models_final/summarizer-model-1_epoch.pt'

def initialize(MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT):
    '''
    function to read, transform, load data in tensor format along with initializing the model for training
    args : MAX_VOCAB - maximum vocabulary size
           BATCH_SIZE - number of datasamples in a batch
           ENC_EMB_DIM - dimension of the encoder
           DEC_EMB_DIM - dimension of the decoder
           ENC_HID_DIM - hidden dimension of the encoder
           DEC_HID_DIM - hidden dimesion of the decoder
           ENC_DROPOUT - encoder dropout
           DEC_DROPOUT - decoder dropout
    returns : model - initialized model
              train_iterator - training data in tensor format
              val_iterator - validation data in tensor format
              test_iterator - test data in tensor format
              optimizer - initialized optimizer
              criterion- loss function
              device - gpu/cpu
              SRC - source vocabulary
              TRG - target vocabulary
    '''
    train_source_txt = read_data("./data/train.txt.src")
    train_target_txt = read_data("./data/train.txt.tgt")
    val_source_txt = read_data("./data/val.txt.src")
    val_target_txt = read_data("./data/val.txt.tgt")
    test_source_txt = read_data("./data/test.txt.src")
    test_target_txt = read_data("./data/test.txt.tgt")
    processed_train_source_txt = preprocess(train_source_txt)
    processed_train_target_txt = preprocess(train_target_txt)
    processed_val_source_txt = preprocess(val_source_txt)
    processed_val_target_txt = preprocess(val_target_txt)
    processed_test_source_txt = preprocess(test_source_txt)
    processed_test_target_txt = preprocess(test_target_txt)
    SRC, TRG = create_vocab(MAX_VOCAB,processed_train_source_txt, processed_train_target_txt)

    train_iterator = data_loader(BATCH_SIZE, processed_train_source_txt, processed_train_target_txt, SRC, TRG)
    val_iterator = data_loader(BATCH_SIZE, processed_val_source_txt,processed_val_target_txt, SRC, TRG)
    test_iterator = data_loader(BATCH_SIZE, processed_test_source_txt,processed_test_target_txt, SRC, TRG)

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    print(len(SRC.vocab))
    model = load_model(INPUT_DIM, OUTPUT_DIM,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT,SRC_PAD_IDX,device)
    optimizer = optim.Adam(model.parameters())
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

    best_valid_loss = float('inf')
    return model, train_iterator, val_iterator, test_iterator, optimizer, criterion, device , SRC, TRG



def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT):
     '''
    function to train the model
    args : MAX_VOCAB - maximum vocabulary size
           BATCH_SIZE - number of datasamples in a batch
           ENC_EMB_DIM - dimension of the encoder
           DEC_EMB_DIM - dimension of the decoder
           ENC_HID_DIM - hidden dimension of the encoder
           DEC_HID_DIM - hidden dimesion of the decoder
           ENC_DROPOUT - encoder dropout
           DEC_DROPOUT - decoder dropout
    '''
    print("Training Started..............................")
    model, train_iterator, val_iterator, test_iterator, optimizer, criterion, device, SRC, TRG=initialize(MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT)
    for epoch in tqdm(range(N_EPOCHS)):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP,device)
        end1_time = time.time()
        valid_loss = evaluate(model, val_iterator, criterion, device)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        torch.save(model.state_dict(), '/models/summarizer-model-'+str(epoch)+'_epoch.pt')
        # if valid_loss < best_valid_loss:
        #     best_valid_loss = valid_loss
        #     torch.save(model.state_dict(), 'tut4-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load('./models/summarizer-model-1_epoch.pt'))
    test_loss = evaluate(model, test_iterator, criterion,device)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    print("Training Completed..............................")


def predict_summary(model_path,MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT):
     '''
    function to read, transform, load data in tensor format along with initializing the model for training
    args : model_path - path of the model checkpoint
           MAX_VOCAB - maximum vocabulary size
           BATCH_SIZE - number of datasamples in a batch
           ENC_EMB_DIM - dimension of the encoder
           DEC_EMB_DIM - dimension of the decoder
           ENC_HID_DIM - hidden dimension of the encoder
           DEC_HID_DIM - hidden dimesion of the decoder
           ENC_DROPOUT - encoder dropout
           DEC_DROPOUT - decoder dropout
    '''
    model, train_iterator, val_iterator, test_iterator, optimizer, criterion, device, SRC, TRG=initialize(MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT)
    model.load_state_dict(torch.load(model_path))
    test_loss = evaluate(model, test_iterator, criterion,device)
    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
    test_source_txt = read_data("../data/test.txt.src")
    test_target_txt = read_data("../data/test.txt.tgt")
    processed_test_source_txt = preprocess(test_source_txt)
    processed_test_target_txt = preprocess(test_target_txt)

    summarized_list=[]
    for src in tqdm(processed_test_source_txt):
        translation, attention = translate_sentence(src, SRC, TRG, model, device,summarization_lenth)
        summarized_list.append(translation)
    summarized_strings = [(" ").join(row) for row in summarized_list] 
    predicted_df = pd.DataFrame(summarized_strings)
    predicted_df.to_csv("summarized_text.csv",index=False,Header=False)
    bleu_score = calculate_bleu(processed_test_source_txt, processed_test_target_txt, SRC, TRG, model, device)
    print(f'BLEU score = {bleu_score*100:.2f}')

predict_summary(model_path,MAX_VOCAB,BATCH_SIZE,ENC_EMB_DIM,DEC_EMB_DIM,ENC_HID_DIM,DEC_HID_DIM,ENC_DROPOUT,DEC_DROPOUT)
