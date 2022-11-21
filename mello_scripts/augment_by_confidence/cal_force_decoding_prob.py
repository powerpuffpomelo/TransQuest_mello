# 调用外部翻译模型，计算force decoding概率
import os
import torch

year = '21'
split = 'test'
data_prefix = '/opt/tiger/fake_arnold/qe_data/qe_data_mello/' + split + year + '/en-de-' + split + year + '/'
src_path = data_prefix + split + year + '.src'
mt_path = data_prefix + split + year + '.mt'
prob_save_prefix = '/opt/tiger/fake_arnold/TransQuest_mello/checkpoints/translation_prob/t5_large/'
prob_save_path = prob_save_prefix + 'prob_' + split + year + '.txt'
if not os.path.exists(prob_save_prefix):
    os.makedirs(prob_save_prefix, exist_ok=True)

# ================================== T5 ================================== #
from transformers import T5Tokenizer, T5ForConditionalGeneration

mname = '/opt/tiger/fake_arnold/TransQuest_mello/transformers/t5-large'
tokenizer = T5Tokenizer.from_pretrained(mname)
model = T5ForConditionalGeneration.from_pretrained(mname)
# tokenizer.save_pretrained('/opt/tiger/fake_arnold/TransQuest_mello/transformers/t5-large')
# model.save_pretrained('/opt/tiger/fake_arnold/TransQuest_mello/transformers/t5-large')

device = 'cuda:0'
model.to(device)

with open(src_path, 'r', encoding='utf-8') as fsrc, \
        open(mt_path, 'r', encoding='utf-8') as fmt, \
        open(prob_save_path, 'w', encoding='utf-8') as fsave:
    src_lines = fsrc.readlines()
    mt_lines = fmt.readlines()
    for src_line, mt_line in zip(src_lines, mt_lines):
        src_line = src_line.strip('\n')
        mt_line = mt_line.strip('\n')
        src_line_for_t5 = 'translate English to German: ' + src_line
        token2word_tag_list = []
        for word in mt_line.split():
            tokens = tokenizer.encode(word, add_special_tokens=False)
            token2word_tag_list.extend([1] + [0] * (len(tokens) - 1))  # 首token1, 再而0

        with torch.no_grad():
            encoded_src = tokenizer(src_line_for_t5, return_tensors='pt')
            encoded_mt = tokenizer(mt_line, return_tensors='pt', add_special_tokens=False)
            src_tokens = encoded_src['input_ids'].to(device)
            # src_mask = encoded_src['attention_mask'].to(device)
            mt_tokens = encoded_mt['input_ids'].to(device)
            # mt_mask = encoded_mt['attention_mask'].to(device)

            output = model(
                input_ids=src_tokens,
                # attention_mask=src_mask,
                labels=mt_tokens
            )

            logits = output.logits.view(-1, model.config.vocab_size)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            probs_token = probs.gather(dim=-1, index=mt_tokens.view(-1, 1)).squeeze().tolist()
            probs_word = [probs_token[i] for i in range(len(probs_token)) if token2word_tag_list[i] == 1]
            fsave.write(' '.join(map(str, probs_word)) + '\n')


# python3 mello_scripts/augment_by_confidence/cal_force_decoding_prob.py