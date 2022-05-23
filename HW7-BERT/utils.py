import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset 
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast, RobertaTokenizer
from transformers import AutoTokenizer, AutoModelWithLMHead
  

from tqdm.auto import tqdm


def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]

def pre_tokenize(start, end, token):
    new_start, new_end = 512, 512
    start_index, end_index = 0, 0
    count = 0
    for i, tokens in enumerate(token):
        if tokens == '[UNK]' or tokens == '[SEP]' or tokens == '[CLS]':
            if i == start:
                new_start = count
            if i == end:
                new_end = count
            count += 1
        else:
            for j in tokens:
                if i == start and start_index == 0:
                    new_start = count
                    start_index = 1
                if i == end:
                    new_end = count
                    end_index = 1
                if j != '#':
                    count += 1
    return new_start, new_end 

device = "cuda" if torch.cuda.is_available() else "cpu"
def evaluate(data, output, tokenizer, paragraph=None, tokenized_paragraph=None):
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing 
    # Hint: Open your prediction file to see what is wrong 
    
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    stride = 300 # self.doc_stride
    start_index_all, end_index_all = 0, 0

    for k in range(num_of_windows):
        mask = data[1][0][k].bool() & data[2][0][k].bool()
        mask_start = torch.masked_select(output.start_logits[k].to(device), mask.to(device))[:-1].to(device)
        start_prob, start_index = torch.max(mask_start, dim=0)
        mask_end = torch.masked_select(output.end_logits[k].to(device), mask.to(device))[start_index:-1].to(device)
        end_prob, end_index = torch.max(mask_end, dim=0)
        end_index += start_index

        # if end_index < start_index or data[0][0][k][start_index].item() == 0:
        #     continue
        # Probability of answer is calculated as sum of start_prob and end_prob
        prob = start_prob + end_prob
        masked_data = torch.masked_select(data[0][0][k], mask)[:-1].to(device)
        # Replace answer if calculated probability is larger than previous windows
        if prob > max_prob and end_index > start_index:
            max_prob = prob
            start_index_all = start_index.item() + stride * k
            end_index_all = end_index.item() + stride * k
            # Convert tokens to chars (e.g. [1920, 7032] --> "大 金")
            answer = tokenizer.decode(masked_data[start_index : end_index + 1])
    # Post-Processing with [UNK]
    if '[UNK]' in answer:
        print("Original answer is ", answer)
        new_start, new_end = pre_tokenize(start_index_all, end_index_all, tokenized_paragraph)
        answer = paragraph[new_start : new_end + 1]
        print('Modified answer is ', answer.replace(" ", ""))
    # Remove spaces in answer (e.g. "大 金" --> "大金")
    return answer.replace(" ", "")