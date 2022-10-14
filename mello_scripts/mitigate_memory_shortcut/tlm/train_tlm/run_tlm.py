import argparse
import os
import pandas as pd
import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    XLMRobertaForMaskedLM,
)
from transformers.optimization import (
    AdamW,
    Adafactor,
    get_linear_schedule_with_warmup,
)


from dataset import prepare_dataloader, Dataset, tlm_collator


def parse_args():
    parser = argparse.ArgumentParser(description="train a transformers model on TLM task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--train_src", type=str, default=None, help="train file"
    )
    parser.add_argument(
        "--train_tgt", type=str, default=None, help="train file"
    )
    parser.add_argument(
        "--val_src", type=str, default=None, help="validation file"
    )
    parser.add_argument(
        "--val_tgt", type=str, default=None, help="validation file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=2,
    )
    parser.add_argument(
        "--optimizer", type=str, default='AdamW',
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5,
    )
    parser.add_argument(
        "--max_epochs", type=int, default=1,
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0,
    )
    parser.add_argument(
        "--device_id", type=int, default=0,
    )
    parser.add_argument(
        "--eval_steps", type=int, default=50,
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Where to store the final model."
    )
    parser.add_argument(
        "--save_name", type=str, default=None,
    )
    args = parser.parse_args()

    return args


def save_pretrained(model, step, save_name, args):
    save_path = args.output_dir + "/{}_{}".format(save_name, step)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    model.save_pretrained(save_path)
    print("model saved in {}".format(save_path))


def main():
    args = parse_args()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    train_dataloader = prepare_dataloader(args.train_src, args.train_tgt, args)
    #validation_data = prepare_dataloader(args.val_src, args.val_tgt, args)

    grad_accumulation = args.gradient_accumulation_steps
    t_total = len(train_dataloader) // grad_accumulation * args.max_epochs

    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

    model = XLMRobertaForMaskedLM.from_pretrained(args.model_name_or_path)
    model.to(device)
    
    if args.optimizer == 'AdamW':
        optimizer = AdamW(
            params=model.parameters(),
            lr=args.learning_rate,
        )
    else:
        optimizer = Adafactor(
            params=model.parameters(),
            lr=args.learning_rate,
        )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total,
    )

    cur_epoch = 1
    cur_step = 0

    while True:
        cur_epoch += 1
        epoch_loss = 0.
        model.train()
        
        with tqdm(total = len(train_dataloader), ncols=200) as pbar:
            for batch_num, batch_data in enumerate(train_dataloader):
                batch_input, batch_label = batch_data

                output = model(
                    input_ids=batch_input['input_ids'].to(device),
                    attention_mask=batch_input['attention_mask'].to(device),
                    labels=batch_label.to(device)
                )
                cur_step += 1

                loss = output.loss
                loss = loss / grad_accumulation

                epoch_loss += loss.item()
                loss.backward()
                if (batch_num + 1) % grad_accumulation == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad() 

                pbar.set_postfix({'cur_loss':'{:.4f}'.format(loss.item()), 'total_loss':'{:.4f}'.format(epoch_loss/(batch_num + 1))})
                pbar.update(1)

                if cur_step % args.eval_steps == 0:
                    model.eval()
                    # TODO eval
                    model.train()
                    save_pretrained(model, cur_step, args.save_name, args)

                if cur_step >= t_total * grad_accumulation:
                    break
        
        if cur_step >= t_total * grad_accumulation:
            break

if __name__ == "__main__":
    main()