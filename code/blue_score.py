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
        
        pred_trg, _ = translate_sentence(src, src_field, trg_field, model, device, max_len)
        
        #cut off <eos> token
        pred_trg = pred_trg[:-1]
        
        pred_trgs.append(pred_trg)
        trgs.append([trg])
        
    return bleu_score(pred_trgs, trgs)