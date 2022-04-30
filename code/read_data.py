from tqdm import tqdm
def read_data(filename):
    '''
    function to read data from the given input path
    args: path - path of the input file
    returns : lines - list of lines in the provided file
    '''
    with open(filename) as f:
        lines = f.readlines()
    return lines

def preprocess(file):
    '''
    function to preprocess the data
    args : data - list of lines in the file
    returns : list of lines with "start" and "end" token appended to it.
    '''
    pre_processed_file=[]
    for line in tqdm(file):
        pre_processed_file.append(['<sos>']+line.split()+['<eos>'])
    return pre_processed_file