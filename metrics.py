from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from sacrebleu.metrics import BLEU
import pandas as pd
import torch
import argparse

parser = argparse.ArgumentParser('Metrics')
parser.add_argument('-device', default='cuda:2', type=str, help='device_of_model_calculation')
parser.add_argument('-target', default='data/detox/test_ru_RU.0', type=str, help='target file path')
parser.add_argument('-predict', default='data/outputs/mbart_en_data_ru_RU.0', type=str, help='prediction file path')
parser.add_argument('-df_res', default='result/df_res.csv', type=str, help='path to df with results')
parser.add_argument('-ppl_model_name', default='sberbank-ai/rugpt2large', type=str, help='model name from ppl')
parser.add_argument('-model_name', default='', type=str, help='name of model to log it in df_res')
parser.add_argument('-exper_params', default='sberbank-ai/rugpt2large', type=str, help='experiment params to log it in df_res')
opt = parser.parse_args()

device = opt.device

# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# pred_path = 'data/outputs/mbart_en_data_ru_RU.0'
# target_path = 'data/detox/test_ru_RU.0'
# df_res_path = 'result/df_res.csv'
# model_name_or_path = "sberbank-ai/rugpt2large"


def load_data(pred_path, target_path):
    targets = []
    with open(target_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            targets.append(line)

    preds = []
    with open(pred_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            preds.append(line)

    return targets, preds


def transform_text(text):
    text = text.strip().replace('"', '').replace("'", "")
    text = '\n' + text + '.\n'
    return text


def calculate_perplexity(preds, targets, model=None, tokenizer=None,
                         model_name_or_path="sberbank-ai/rugpt2large", weight=True, and_diff=False):
    if model is None:
        print('loading model')
        model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)
    if tokenizer is None:
        print('loading tokenizer')
        tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

    mean_ppl = 0
    len_tockens = 0
    if and_diff:
        mean_ppl_diff = 0

    if len(preds) != len(targets):
        raise ValueError("Number of predicted texts should be equal number of original texts")

    for i in tqdm(range(len(preds))):

        pred, target = transform_text(preds[i]), transform_text(targets[i])

        input_ids_pred = tokenizer.encode(pred, return_tensors="pt").to(device)
        out_pred = model(input_ids_pred.to(device), labels=input_ids_pred.to(device))
        loss_pred = out_pred.loss.cpu().detach().numpy().tolist()

        input_ids_target = tokenizer.encode(pred, return_tensors="pt").to(device)
        out_target = model(input_ids_target.to(device), labels=input_ids_target.to(device))
        loss_target = out_target.loss.cpu().detach().numpy().tolist()

        if weight:
            len_tok = len(input_ids_target.flatten()) - 1
            mean_ppl += (loss_pred) * len_tok
            if and_diff:
                mean_ppl_diff += (loss_target - loss_pred) * len_tok
            len_tockens += len_tok
        else:
            mean_ppl += loss_target - loss_pred
            if and_diff:
                mean_ppl_diff += (loss_target - loss_pred)
            len_tockens += 1

    mean_ppl /= len_tockens

    if and_diff:
        mean_ppl_diff /= len_tockens
        return mean_ppl, mean_ppl_diff
    else:
        return mean_ppl


def calculate_metrics(targets, preds, ppl_model_name="sberbank-ai/rugpt2large"):
    model_ppl = GPT2LMHeadModel.from_pretrained(ppl_model_name).to(device)
    tokenizer_ppl = GPT2Tokenizer.from_pretrained(ppl_model_name)

    sta = 0  # TODO
    bleu = BLEU(force=True).corpus_score(preds, targets).score
    ppl, ppl_diff = calculate_perplexity(preds, targets, model=model_ppl,
                                         tokenizer=tokenizer_ppl, weight=True, and_diff=True)

    return sta, bleu, ppl, ppl_diff


def main_metrics(pred_path, target_path, df_res_path, model_name='', exp_details='',
                 ppl_model_name="sberbank-ai/rugpt2large", verbose=True):
    df_res = pd.read_csv(df_res_path)
    targets, preds = load_data(pred_path, target_path)
    sta, bleu, ppl, ppl_diff = calculate_metrics(targets, preds, ppl_model_name)

    new_line = dict(zip(df_res.columns, [model_name, exp_details, sta, bleu, ppl, ppl_diff]))

    df_res.append(new_line, ignore_index=True)
    df_res.to_csv('result/df_res.csv', index=None)

    if verbose:
        print(new_line)


if __name__ == '__main__':
    # main_metrics(pred_path, target_path, df_res_path)
    main_metrics(opt.predict, opt.target, opt.df_res, ppl_model_name=opt.ppl_model_name,
                 model_name=opt.model_name, exp_details=opt.exper_params)
