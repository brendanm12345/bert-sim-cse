import random
import numpy as np
import argparse
from types import SimpleNamespace
import datetime

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
    load_multitask_and_simcse_data,
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
    def __init__(self, temperature=0.05):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        # Normalize the embeddings to have unit length
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Calculate cosine similarity matrix (size: batch_size x batch_size)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # For numerical stability, subtract the maximum value in each row
        max_vals = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        similarity_matrix = similarity_matrix - max_vals.detach()
        
        # The diagonal entries are the similarities between each embedding and itself
        # We want to exclude these when calculating the denominator in softmax
        logits_mask = torch.eye(similarity_matrix.size(0)).bool().to(embeddings.device)
        similarity_matrix.masked_fill_(logits_mask, float('-inf'))
        
        # Calculate softmax along each row, but exclude the diagonal (self-similarity)
        softmax_scores = F.softmax(similarity_matrix, dim=1)
        
        # The targets are the positions of the positive examples in the softmax_scores matrix
        # For unsupervised SimCSE, the positive example for each anchor is its pair (next in sequence)
        targets = torch.arange(embeddings.size(0)).to(embeddings.device)
        if embeddings.size(0) % 2 == 0:  # Assuming even batch size for simplicity
            targets = (targets + 1) - 2 * (targets % 2)
        
        # Calculate the log likelihood of the positive examples
        log_probs = torch.log(torch.gather(softmax_scores, 1, targets.unsqueeze(1)).squeeze(1))
        
        # The loss is the negative log likelihood averaged across the batch
        loss = -log_probs.mean()
        print("this is loss", loss)
        return loss


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

        self.sentiment_classifier = nn.Linear(
            BERT_HIDDEN_SIZE, N_SENTIMENT_CLASSES)
        # * 2 for concat sentence embeddings
        self.paraphrase_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)
        # * 2 for concat sentence embeddings
        self.similarity_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # SIMCSE: Added code
        self.simcse = True
        self.contrastive_classifier = nn.Linear(BERT_HIDDEN_SIZE * 2, 1)

    # SIMCSE: Updated this function
    def forward(self, input_ids, attention_mask, simcse=False):
        # Process input through BERT - sentence data ids gets turned to embeddings
  
        

        if simcse:
            # For SimCSE, apply dropout twice to get two embeddings for the same input

            output_first = self.bert(input_ids, attention_mask=attention_mask)
            output_first_pooled = output_first["pooler_output"]

            output_second = self.bert(input_ids, attention_mask=attention_mask)
            output_second_pooled = output_second["pooler_output"]

            return output_first_pooled, output_second_pooled
        else:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
            return outputs["pooler_output"]

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        embeddings = self.forward(input_ids, attention_mask)
        embeddings = self.dropout(embeddings)
        logits = self.sentiment_classifier(embeddings)
        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.paraphrase_classifier(embeddings)
        return logits

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        embeddings = torch.cat((embeddings_1, embeddings_2), dim=1)
        embeddings = self.dropout(embeddings)
        logits = self.similarity_classifier(embeddings)
        return logits


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


def train_all(x):
    '''Train MultitaskBERT.
    SST, Paraphrase, STS, SimCSE
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data, contrastive_train_data = load_multitask_and_simcse_data(
        args.sst_train, args.para_train, args.sts_train, args.simcse_train, split='train')

    sst_dev_data, num_labels, para_dev_data, sts_dev_data, contrastive_train_data = load_multitask_and_simcse_data(
        args.sst_dev, args.para_dev, args.sts_dev, args.simcse_train, split='dev')

    # SIMCSE: Load SimCSE training data
    contrastive_train_data = SimCSEDataset(args.simcse_train, args)

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_train_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_train_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_dataloader = DataLoader(
        para_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(
        para_dev_data, shuffle=True, batch_size=args.batch_size, collate_fn=para_train_data.collate_fn)

    sts_train_dataloader = DataLoader(
        sts_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(
        sts_dev_data, shuffle=True, batch_size=args.batch_size, collate_fn=sts_train_data.collate_fn)
    contrastive_loss_fn = ContrastiveLoss().to(device)

    # SIMCSE: Create DataLoader for SimCSE training data
    contrastive_train_dataloader = DataLoader(
        contrastive_train_data, shuffle=True, batch_size=args.batch_size, collate_fn=contrastive_train_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)

    # Added
    best_dev_sentiment_accuracy = 0
    best_dev_paraphrase_accuracy = 0
    best_dev_sts_corr = -1

    # for each epoch
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)
        # SIMCSE
        contrastive_iter = iter(contrastive_train_dataloader)

        max_steps = max(len(sst_train_dataloader), len(
            para_train_dataloader), len(sts_train_dataloader), len(contrastive_train_dataloader))

        # for each batch
        total_contrastive_loss_for_epoch = 0
        for _ in tqdm(range(max_steps), desc=f"Epoch {epoch}"):
            # train simcse
            try:
                simcse_batch = next(contrastive_iter)
                input_ids, attention_mask = simcse_batch['input_ids'].to(
                    device), simcse_batch['attention_mask'].to(device)
                optimizer.zero_grad()
                # Forward pass through the model to get two sets of embeddings
                pooled_output_first, pooled_output_second = model(
                    input_ids=input_ids, attention_mask=attention_mask, simcse=True)
                
                # Concatenate the embeddings from the two passes to form a single batch
                embeddings = torch.cat((pooled_output_first, pooled_output_second), dim=0)

                # Calculate contrastive loss
                loss = contrastive_loss_fn(embeddings)
                loss.backward()
                optimizer.step()
                # save contrastive loss specifically so we can track it at end of each epoch
                contrastive_loss = loss.item()
                total_contrastive_loss_for_epoch += contrastive_loss
                print("contrastive loss: ", contrastive_loss)
                train_loss += loss.item()
                num_batches += 1
            except StopIteration:
                pass

            # train sst
            try:
                batch = next(sst_iter)
                b_ids, b_mask, b_labels = (batch['token_ids'],
                                           batch['attention_mask'], batch['labels'])
                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)
                b_labels = b_labels.to(device)
                optimizer.zero_grad()
                logits = model.predict_sentiment(b_ids, b_mask)
                loss = F.cross_entropy(
                    logits, b_labels.view(-1), reduction='mean')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            except StopIteration:
                pass

            # train para
            try:
                batch = next(para_iter)
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                                  batch['attention_mask_1'],
                                                                  batch['token_ids_2'],
                                                                  batch['attention_mask_2'],
                                                                  batch['labels'])
                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)
                optimizer.zero_grad()
                logits = model.predict_paraphrase(
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                loss = F.binary_cross_entropy_with_logits(
                    logits, b_labels.view(-1, 1).float(), reduction='mean')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            except StopIteration:
                pass

            # train sts
            try:
                batch = next(sts_iter)
                b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                                  batch['attention_mask_1'],
                                                                  batch['token_ids_2'],
                                                                  batch['attention_mask_2'],
                                                                  batch['labels'])
                b_ids_1 = b_ids_1.to(device)
                b_mask_1 = b_mask_1.to(device)
                b_ids_2 = b_ids_2.to(device)
                b_mask_2 = b_mask_2.to(device)
                b_labels = b_labels.to(device)
                optimizer.zero_grad()
                logits = model.predict_similarity(
                    b_ids_1, b_mask_1, b_ids_2, b_mask_2)
                b_labels = b_labels.float()
                loss = F.mse_loss(logits, b_labels.view(-1, 1),
                                  reduction='mean')
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                num_batches += 1
            except StopIteration:
                pass

        train_loss = train_loss / (num_batches)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        improvement = False
        if dev_sentiment_accuracy > best_dev_sentiment_accuracy:
            best_dev_sentiment_accuracy = dev_sentiment_accuracy
            improvement = True
        if dev_paraphrase_accuracy > best_dev_paraphrase_accuracy:
            best_dev_paraphrase_accuracy = dev_paraphrase_accuracy
            improvement = True
        if dev_sts_corr > best_dev_sts_corr:
            best_dev_sts_corr = dev_sts_corr
            improvement = True

        if improvement:
            save_model(model, optimizer, args, config, args.filepath)
            print(f"Improvement! Model saved at epoch {epoch}")
            print(f"- Sentiment acc: {best_dev_sentiment_accuracy:.3f}")
            print(f"- Paraphrase acc: {best_dev_paraphrase_accuracy:.3f}")
            print(f"- STS corr: {best_dev_sts_corr:.3f}")

        print(f"Epoch {epoch} Evaluation:")
        print(f"- Total contrastive loss: {total_contrastive_loss_for_epoch:.3f}")
        print(f"- Sentiment acc: {dev_sentiment_accuracy:.3f}")
        print(f"- Paraphrase acc: {dev_paraphrase_accuracy:.3f}")
        print(f"- STS corr: {dev_sts_corr:.3f}")


def test_all(args):
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

    size = "small"

    parser.add_argument("--simcse_train", type=str,
                        default="data/{size}/unsup_simcse.csv")

    # Updated file paths to shortened data versions
    parser.add_argument("--sst_train", type=str,
                        default=f"data/{size}/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default=f"data/{size}/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str,
                        default=f"data/{size}/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str,
                        default=f"data/{size}/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default=f"data/{size}/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default=f"data/{size}/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default=f"data/{size}/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default=f"data/{size}/sts-dev.csv")
    parser.add_argument("--sts_test", type=str,
                        default=f"data/{size}/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    parser.add_argument("--sst_dev_out", type=str,
                    default=f"predictions/sst-dev-output-{current_datetime}.csv")
    parser.add_argument("--sst_test_out", type=str,
                        default=f"predictions/sst-test-output-{current_datetime}.csv")

    parser.add_argument("--para_dev_out", type=str,
                        default=f"predictions/para-dev-output-{current_datetime}.csv")
    parser.add_argument("--para_test_out", type=str,
                        default=f"predictions/para-test-output-{current_datetime}.csv")

    parser.add_argument("--sts_dev_out", type=str,
                        default=f"predictions/sts-dev-output-{current_datetime}.csv")
    parser.add_argument("--sts_test_out", type=str,
                        default=f"predictions/sts-test-output-{current_datetime}.csv")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Save path.
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-{current_datetime}-multitask.pt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_all(args)
    test_all(args)

