import os
import torch
import transformers


class BertModel(object):
    def __init__(self, name='BERT', model_path=None, vocab_file=None):
        self.name_ = name
        self.model_path = model_path

        model = transformers.BertForMaskedLM.from_pretrained(self.model_path)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        self.model_ = model

        self.tokenizer_ = transformers.BertTokenizer(vocab_file=vocab_file,
                                                     do_lower_case=False)
        self.alphabet_ = self.tokenizer_.vocab
        self.unk_idx_ = self.tokenizer_.vocab['[UNK]']

        self.vocabulary_ = {
            tok: self.alphabet_[tok]
            for tok in self.alphabet_ if '[' not in tok
        }
