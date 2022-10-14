import pandas as pd
import random
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaTokenizer,
    DataCollatorForLanguageModeling,
)


class tlm_collator():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.random = random.Random(666)

    def __call__(self, data):
        batch_sample = self.tokenizer(data, padding='max_length', truncation=True, max_length=512)

        input_ids_list = []
        label_list = []
        attention_mask_list = []

        for idx in range(len(batch_sample['input_ids'])):
            sample = batch_sample['input_ids'][idx]
            _label = []
            _input_id = []
            for token_id in sample:
                if token_id is not self.tokenizer.pad_token_id:
                    if self.random.random() < 0.15:
                        _label.append(token_id)
                        _input_id.append(self.tokenizer.mask_token_id)
                    else:
                        _label.append(-100)
                        _input_id.append(token_id)
                else:
                    _label.append(-100)
                    _input_id.append(token_id)

            input_ids_list.append(_input_id)
            label_list.append(_label)
            attention_mask_list.append(batch_sample['attention_mask'][idx])     

        batch_input = {
            'input_ids': torch.tensor(input_ids_list),
            'attention_mask': torch.tensor(attention_mask_list)
        }
        batch_label = torch.tensor(label_list)

        return batch_input, batch_label


class Dataset(Data.Dataset):
    def __init__(self, src_file, tgt_file) -> None:
        data = []
        with open(src_file, 'r', encoding='utf-8') as fsrc, open(tgt_file, 'r', encoding='utf-8') as ftgt:
            for src_line, tgt_line in zip(fsrc.readlines(), ftgt.readlines()):
                data.append([src_line.strip('\n'), tgt_line.strip('\n')])
        self.data = pd.DataFrame(data, columns=['src', 'tgt'])
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def prepare_dataloader(src_file, tgt_file, args):
    dataset = Dataset(src_file, tgt_file)
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name_or_path)
    collate_fn = tlm_collator(tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader