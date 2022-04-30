from torchtext.legacy import data
from tqdm import tqdm
def create_vocab(MAX_VOCAB,processed_train_source_txt, processed_train_target_txt):
    '''
    function to create source vocabulary and target vocabulary from the data for training
    args : MAX_VOCAB - maximum vocabulary size during training
           processed_train_source_txt - preprocessed source training dataset
           processed_train_target_txt - preprocessed target training dataset
    reurns : SRC - source vocabulary , target vocabulary
    '''
    SRC = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')
    TRG = data.Field(tokenize = 'spacy',tokenizer_language = 'en_core_web_sm')
    SRC.build_vocab([i for i in tqdm(processed_train_source_txt)], max_size = MAX_VOCAB)
    TRG.build_vocab([i for i in tqdm(processed_train_target_txt)], max_size = MAX_VOCAB)
    return SRC, TRG