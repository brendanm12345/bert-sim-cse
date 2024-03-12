import random
import numpy as np
import argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
    SimCSEDataset,
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask


TQDM_DISABLE = False


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, positive_pairs=None):
        super(ContrastiveLoss, self).__init__()
        self.positive_pairs = positive_pairs
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self):
        total_loss = 0.0
        for anchor, positive in self.positive_pairs:
            # Calculate the numerator of the softmax for the positive pair
            positive_similarity = torch.exp(
                torch.dot(anchor, positive) / self.temperature)

            # Calculate the denominator of the softmax for all negative pairs
            negative_sum = positive_similarity  # include the positive pair in the denominator
            for i in range(len(self.positive_pairs)):
                # negative_similarity = torch.exp(torch.dot(anchor, negative_pairs[i][1]) / tau)
                # negative_sum += coefficient_1 * negative_similarity
                positive_similarity = torch.exp(
                    torch.dot(anchor, self.positive_pairs[i][1]) / self.temperature)
                negative_sum += positive_similarity

            # Compute the loss for the current positive pair
            pair_loss = -torch.log(positive_similarity / negative_sum)
            total_loss += pair_loss

        return total_loss


class GaussianDropout(nn.Module):
    def __init__(self, p=0.1):
        super(GaussianDropout, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # Sample noise
            noise = torch.randn_like(x) * self.p
            return x + noise
        return x


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # TODO Added code here
        self.sentiment_classifier = nn.Linear(
            BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # * 2 for concat sentence embeddings
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        # * 2 for concat sentence embeddings
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # SIMCSE: Added code
        self.simcse = config.get('simcse', False)
        self.contrastive_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

    # SIMCSE: Updated this function
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Process input through BERT
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        if self.simcse:
            # For SimCSE, apply dropout twice to get two embeddings for the same input
            pooled_output = outputs.pooler_output
            # Apply dropout first time
            pooled_output_first = self.dropout(pooled_output)
            # Re-apply dropout to get a second, different embedding
            pooled_output_second = self.dropout(pooled_output)

            return pooled_output_first, pooled_output_second
        else:
            # Original processing
            return outputs.pooler_output


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_simcse(x):
    '''Train MultitaskBERT.
    Currently computes and combines loss for each task during each iteration to optimzie for all three. Need to migrate to SimCSE contrastive learning.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Load SimCSE training and development data
    simcse_train_data = SimCSEDataset(args.simcse_train)
    simcse_dev_data = SimCSEDataset(args.simcse_dev)

    # Create DataLoaders for SimCSE training and development data
    simcse_train_dataloader = DataLoader(
        simcse_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=simcse_train_data.collate_fn)
    simcse_dev_dataloader = DataLoader(
        simcse_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=simcse_dev_data.collate_fn)

    config = SimpleNamespace(hidden_dropout_prob=args.hidden_dropout_prob,
                             num_labels=2,  # Assuming binary classification for simplicity < this might be wrong since we want to test on same tasks
                             hidden_size=768,
                             data_dir='.',
                             option=args.option,
                             simcse=True)

    model = MultitaskBERT(config).to(device)
    contrastive_loss_fn = ContrastiveLoss().to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(simcse_train_dataloader, desc=f"Epoch {epoch}"):
            input_ids, attention_mask = batch['input_ids'].to(
                device), batch['attention_mask'].to(device)

            # Forward pass to get embeddings. For SimCSE, we expect two sets of embeddings per input due to dropout variation.
            embeddings1, embeddings2 = model(input_ids, attention_mask)

            positive_pairs = list(zip(embeddings1, embeddings2))
            loss = contrastive_loss_fn(positive_pairs=positive_pairs)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(simcse_train_dataloader)
        print(f"Epoch {epoch}: Average training loss: {avg_loss:.4f}")


def test_simcse(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test,
                                args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev,
                                args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(
            sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(sst_test_dataloader,
                                      para_test_dataloader,
                                      sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--simcse_train", type=str,
                        required=True, help="Path to SimCSE training data")

    parser.add_argument("--sst_train", type=str,
                        default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str,
                        default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str,
                        default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str,
                        default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str,
                        default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str,
                        default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str,
                        default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str,
                        default="predictions/sts-test-output.csv")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Save path.
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_simcse(args)
    test_simcse(args)
