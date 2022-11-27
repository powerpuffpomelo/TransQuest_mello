# 调用xlmr模型，计算最接近的词向量相似度
import os
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch
import torch.nn.functional as F
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', '-v', type=str, default='entropy')
args = parser.parse_args()

year = '21'
split = 'test'
data_prefix = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/' + split + year + '/en-de-' + split + year + '/'
src_path = data_prefix + split + year + '.src'
mt_path = data_prefix + split + year + '.mt'

sim_save_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/embed_sim/'
sim_save_path = sim_save_prefix + 'sim_' + args.version + '_' + split + year + '.txt'
if not os.path.exists(sim_save_prefix):
    os.makedirs(sim_save_prefix, exist_ok=True)

model_path = 'transformers/xlm-roberta-large'
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
model = XLMRobertaModel.from_pretrained(model_path)
# tokenizer.save_pretrained('transformers/xlm-roberta-large')
# model.save_pretrained('transformers/xlm-roberta-large')

device = 'cuda:0'
model.to(device)

with open(src_path, 'r', encoding='utf-8') as fsrc, \
        open(mt_path, 'r', encoding='utf-8') as fmt, \
        open(sim_save_path, 'w', encoding='utf-8') as fs:
    src_lines = fsrc.readlines()
    mt_lines = fmt.readlines()
    id = 0
    for src_line, mt_line in zip(src_lines, mt_lines):
        id += 1
        # print(id)
        if id != 37: continue
        # print(mt_line)
        mt_inputs = tokenizer(mt_line, return_tensors='pt', add_special_tokens=False).to(device)
        mt_outputs = model(**mt_inputs)
        mt_last_hidden_states = mt_outputs.last_hidden_state.squeeze(0)
        
        mt_token_tag = []
        mt_line = mt_line.strip('\n').split()
        for word in mt_line:
            word_token = tokenizer(word, return_tensors='pt', add_special_tokens=False)['input_ids'][0].tolist()
            mt_token_tag.extend([1] + [0] * (len(word_token) - 1))
        # print(mt_token_tag)
        
        src_inputs = tokenizer(src_line, return_tensors='pt', add_special_tokens=False).to(device)
        src_outputs = model(**src_inputs)
        src_last_hidden_states = src_outputs.last_hidden_state.squeeze(0)

        embed_sim_matrix_mt2src = F.softmax(torch.mm(mt_last_hidden_states, src_last_hidden_states.transpose(0, 1)), dim=-1)
        # print(embed_sim_matrix_mt2src)
        # print(F.softmax(torch.cosine_similarity(mt_last_hidden_states.unsqueeze(1), src_last_hidden_states.unsqueeze(0), dim=-1), dim=-1))
        
        # 方案一：sim_max 每个mt token对应的最相似src token的相似度
        if args.version == 'max':
            sim = torch.max(embed_sim_matrix_mt2src, dim=-1).values
        # 方案二：sim_entropy 每个mt token对应的src token相似度的熵
        elif args.version == 'entropy':
            print(embed_sim_matrix_mt2src)
            print(torch.log(embed_sim_matrix_mt2src))
            sim = torch.sum(-embed_sim_matrix_mt2src * torch.log(embed_sim_matrix_mt2src), dim=-1)
        print(sim) # 算出来就是nan
        # print(mt_token_tag)
        # mt word包含的mt token取平均
        sim_list = []
        num = 0
        for i in range(len(mt_token_tag)):
            s = sim[i].item()
            tag = mt_token_tag[i]
            num += 1
            if tag == 1: 
                sim_list.append(s)
            else: 
                sim_list[-1] += s
            if i == len(mt_token_tag) - 1 or mt_token_tag[i + 1] == 1:
                sim_list[-1] /= num
                num = 0

        fs.write(' '.join(list(map(str, sim_list))) + '\n')

# python3 mello_scripts/augment_by_confidence/embed_sim/1_cal_embedding_similarity.py -v entropy