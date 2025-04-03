import re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pickle
tokenizer_name='Rostlab/prot_bert_bfd'
tok = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tok, f)