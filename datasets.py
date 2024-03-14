#!/usr/bin/env python3

'''
This module contains our Dataset classes and functions that load the three datasets
for training and evaluating multitask BERT.

Feel free to edit code in this file if you wish to modify the way in which the data
examples are preprocessed.
'''

import csv

import torch
from torch.utils.data import Dataset
from tokenizer import BertTokenizer
import json
from torch.nn.utils.rnn import pad_sequence


def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


class Supervised_simcse(str):
    def __init__(self, path):
        self.path = path

    def generate_pairs_and_write_to_json(self, file_path, output_json_path):
        pairs = {"positive_pairs": [], "negative_pairs": []}

        with open(file_path, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                if len(row) == 3:  # Ensure row has sentence, entailment, and contradiction
                    sentence, entailment, contradiction = row
                    pairs["positive_pairs"].append(
                        {"sentence": sentence, "entailment": entailment})
                    pairs["negative_pairs"].append(
                        {"sentence": sentence, "contradiction": contradiction})

        with open(output_json_path, mode='w', encoding='utf-8') as json_file:
            json.dump(pairs, json_file, ensure_ascii=False, indent=4)


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):

        sents = [x[0] for x in data]
        labels = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = torch.LongTensor(labels)

        return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(
            all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


# Unlike SentenceClassificationDataset, we do not load labels in SentenceClassificationTestDataset.
class SentenceClassificationTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sents = [x[0] for x in data]
        sent_ids = [x[1] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, all_data):
        token_ids, attention_mask, sents, sent_ids = self.pad_data(all_data)

        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'sents': sents,
            'sent_ids': sent_ids
        }

        return batched_data


class SentencePairDataset(Dataset):
    def __init__(self, dataset, args, isRegression=False):
        self.dataset = dataset
        self.p = args
        self.isRegression = isRegression
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        labels = [x[2] for x in data]
        sent_ids = [x[3] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(
            sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])
        if self.isRegression:
            labels = torch.DoubleTensor(labels)
        else:
            labels = torch.LongTensor(labels)

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                labels, sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         labels, sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'labels': labels,
            'sent_ids': sent_ids
        }

        return batched_data


# Unlike SentencePairDataset, we do not load labels in SentencePairTestDataset.
class SentencePairTestDataset(Dataset):
    def __init__(self, dataset, args):
        self.dataset = dataset
        self.p = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def pad_data(self, data):
        sent1 = [x[0] for x in data]
        sent2 = [x[1] for x in data]
        sent_ids = [x[2] for x in data]

        encoding1 = self.tokenizer(
            sent1, return_tensors='pt', padding=True, truncation=True)
        encoding2 = self.tokenizer(
            sent2, return_tensors='pt', padding=True, truncation=True)

        token_ids = torch.LongTensor(encoding1['input_ids'])
        attention_mask = torch.LongTensor(encoding1['attention_mask'])
        token_type_ids = torch.LongTensor(encoding1['token_type_ids'])

        token_ids2 = torch.LongTensor(encoding2['input_ids'])
        attention_mask2 = torch.LongTensor(encoding2['attention_mask'])
        token_type_ids2 = torch.LongTensor(encoding2['token_type_ids'])

        return (token_ids, token_type_ids, attention_mask,
                token_ids2, token_type_ids2, attention_mask2,
                sent_ids)

    def collate_fn(self, all_data):
        (token_ids, token_type_ids, attention_mask,
         token_ids2, token_type_ids2, attention_mask2,
         sent_ids) = self.pad_data(all_data)

        batched_data = {
            'token_ids_1': token_ids,
            'token_type_ids_1': token_type_ids,
            'attention_mask_1': attention_mask,
            'token_ids_2': token_ids2,
            'token_type_ids_2': token_type_ids2,
            'attention_mask_2': attention_mask2,
            'sent_ids': sent_ids
        }

        return batched_data


class SimCSEDataset(Dataset):
    """
    SIMCSE: Dataset class for SimCSE
    """

    def __init__(self, filepath, args):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentences = self.load_sentences(filepath)

    # every line in the wiki dataset is a sentence
    def load_sentences(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            sentences = [line.strip() for line in file.readlines()]
        return sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        # Tokenize the sentence
        inputs = self.tokenizer(
            sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0)

    def collate_fn(self, batch):
        input_ids = pad_sequence(
            [item[0] for item in batch], batch_first=True, padding_value=0)
        attention_mask = pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


def load_multitask_and_simcse_data(sentiment_filename, paraphrase_filename, similarity_filename, simcse_filename, split='train'):
    print("simcse file name", simcse_filename)
    print("sentiment file name", sentiment_filename)
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label, sent_id))

    print(
        f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(
                                                record['sentence2']),
                                            int(float(record['is_duplicate'])), sent_id))
                except:
                    pass

    print(
        f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']), sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']), sent_id))

    print(
        f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    # Logic to load the SimCSE dataset for unsupervised learning
    simcse_data = []
    with open(simcse_filename, 'r', encoding='utf-8') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent_id = record['id'].lower().strip()
            sentence = preprocess_string(record['sentence'])
            simcse_data.append((sentence, sent_id))

    print(f"Loaded {len(simcse_data)} {split} examples from {simcse_filename}")
    return sentiment_data, num_labels, paraphrase_data, similarity_data, simcse_data


def load_multitask_data(sentiment_filename, paraphrase_filename, similarity_filename, split='train'):
    sentiment_data = []
    num_labels = {}
    if split == 'test':
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                sentiment_data.append((sent, sent_id))
    else:
        with open(sentiment_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent = record['sentence'].lower().strip()
                sent_id = record['id'].lower().strip()
                label = int(record['sentiment'].strip())
                if label not in num_labels:
                    num_labels[label] = len(num_labels)
                sentiment_data.append((sent, label, sent_id))

    print(
        f"Loaded {len(sentiment_data)} {split} examples from {sentiment_filename}")

    paraphrase_data = []
    if split == 'test':
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                paraphrase_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        sent_id))

    else:
        with open(paraphrase_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                try:
                    sent_id = record['id'].lower().strip()
                    paraphrase_data.append((preprocess_string(record['sentence1']),
                                            preprocess_string(
                                                record['sentence2']),
                                            int(float(record['is_duplicate'])), sent_id))
                except:
                    pass

    print(
        f"Loaded {len(paraphrase_data)} {split} examples from {paraphrase_filename}")

    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']), sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']), sent_id))

    print(
        f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return sentiment_data, num_labels, paraphrase_data, similarity_data
