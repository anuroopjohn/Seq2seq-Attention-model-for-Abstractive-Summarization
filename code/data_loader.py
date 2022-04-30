from torch.utils.data import Dataset, DataLoader
import torch
pad_index=1

class SummarizationDataset(Dataset):
    def __init__(self, source,target, src_vocab, TRG_vocab):
        self.X = [torch.LongTensor([src_vocab.get(j,src_vocab['<unk>']) for j in i]) for i in source]
        self.y = [torch.LongTensor([TRG_vocab.get(j,TRG_vocab['<unk>']) for j in i]) for i in target]
        self.lengths = torch.LongTensor([len(i) for i in source])
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self,index):
        return self.X[index],self.y[index], self.lengths[index]

def My_collate(batch):
    '''
    function to customize2 the dataloder class to incorporte batch padding
    args : batch - a single batch of either training/validation/testing dataset.
    returns : source - padded source dataset for a single batch
              target - padded target dataset for a single batch
              lens - length of number of rows in a single batch
    '''
    batch= sorted(batch, key=lambda x:len(x[0]), reverse=True)
    sent_len = [len(row[0]) for row in batch]
    sents = []
    for i in range(len(batch)):
        #print(batch[i][0])
        pad_vector = torch.LongTensor([pad_index]*(max(sent_len)-sent_len[i]))
        sents.append(torch.cat([batch[i][0],pad_vector]))
    source = torch.stack(sents)    
    lens = torch.stack([row[2] for row in batch])
    
    sent_len = [len(row[1]) for row in batch]
    sents = []
    for i in range(len(batch)):
        pad_vector = torch.LongTensor([pad_index]*(max(sent_len)-sent_len[i]))
        sents.append(torch.cat([batch[i][1],pad_vector]))
    target = torch.stack(sents)
    return source.T,target.T,lens

def data_loader(BATCH_SIZE, processed_source_txt, processed_target_txt, SRC, TRG):
    '''
    function that converts raw dataset into pytorch dataloader format
    args :  BATCH_SIZE - number of batches required for training
            processed_train_source_txt - preprocessed source training dataset
            processed_train_target_txt - preprocessed target training dataset
            SRC - source vocabulary
            TRG - target voculary
    returns : train_iterator - training dataset in dataloader format
              test_iterator - testing dataset in dataloader format
              val_iterator -  validation dataset in dataloader format
    '''
    dataset = SummarizationDataset(processed_source_txt,processed_target_txt,SRC.vocab.stoi, TRG.vocab.stoi)
    #val_dataset = SummarizationDataset(processed_val_source_txt,processed_val_target_txt,SRC.vocab.stoi, TRG.vocab.stoi)
    #test_dataset =  SummarizationDataset(processed_test_source_txt,processed_test_target_txt,SRC.vocab.stoi, TRG.vocab.stoi)

    iterator = DataLoader(dataset=dataset, collate_fn=My_collate, batch_size=BATCH_SIZE,shuffle=True)
    # test_iterator = DataLoader(dataset=val_dataset, collate_fn=My_collate, batch_size=BATCH_SIZE,shuffle=True)
    # val_iterator = DataLoader(dataset=test_dataset, collate_fn=My_collate, batch_size=BATCH_SIZE,shuffle=True)
    return iterator