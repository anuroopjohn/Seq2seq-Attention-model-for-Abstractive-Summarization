import spacy
import torch
from torchtext.data.metrics import bleu_score

def summarize_sentence(sentence, src_field, trg_field, model, device, max_len = 20):
    '''
    function to summarize test sentences using the trained model
    args : source - source vocabulary
           target - target vocabulary
           src_field - source field
           trg_field - target field
           model - trained model
           device - cpu/gpu
           max_len - max length of the translated test
    '''
    model.eval()
        
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_len)

    mask = model.create_mask(src_tensor)
        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    attentions = torch.zeros(max_len, 1, len(src_indexes)).to(device)
    
    for i in range(max_len):

        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
                
        with torch.no_grad():
            output, hidden, attention = model.decoder(trg_tensor, hidden, encoder_outputs, mask)

        attentions[i] = attention
            
        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break
    
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:], attentions[:len(trg_tokens)-1]

    from torchtext.data.metrics import bleu_score

def calculate_bleu(source, target, src_field, trg_field, model, device, max_len = 50):
    '''
    function to calculate the Bleu score
    args : source - source vocabulary
           target - target vocabulary
           src_field - source field
           trg_field - target field
           model - trained model
           device - cpu/gpu
           max_len - max length of the translated test
    returns : blue score of the predicton
    '''
    
    trgs = []
    pred_trgs = []
    
    for idx in range(len(source)):
        
        src = source[idx]
        trg = target[idx]
        
        pred_trg, _ = summarize_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)