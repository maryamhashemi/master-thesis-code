import os
from ast import literal_eval
from typing_extensions import Self

import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import CLIPFeatureExtractor


class GenVQADataset(Dataset):
    def __init__(self, tokenizer, dataset_file, img_dir, explode=True, batch_size=32, cat_id = None):
        with open(dataset_file, 'r', encoding='utf-8-sig') as f:
            self.dataset = pd.read_csv(f)

        self.explode=explode
        self.dataset['long_answer'] = self.dataset['long_answer'].apply(literal_eval)
        
        if cat_id is not None:
            self.dataset = self.dataset[self.dataset['category_id']==cat_id]
                   
        if self.explode:
            self.dataset = self.dataset.explode('long_answer')
            
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')

    def __getitem__(self, idx):
        q = self.dataset.iloc[idx]['question']

        img_path = os.path.join(self.img_dir, str(self.dataset.iloc[idx]['image']))

        # with open(img_path, 'r') as f:
        image = Image.open(img_path, mode='r')
        image = self.preprocessor(image, return_tensors='pt')['pixel_values'].squeeze()
            
        # extract sentence data
        tokenized_sentence = self.tokenizer(q)
        input_ids = tokenized_sentence['input_ids']
        attention_mask = tokenized_sentence['attention_mask']


        long_answer = self.dataset.iloc[idx]['long_answer']
        if self.explode:
            tokenized_sentence = self.tokenizer(long_answer)
            label_tokenized = tokenized_sentence['input_ids']
        else:
            label_tokenized = long_answer

        return input_ids, attention_mask, image, label_tokenized
        

    
    def __len__(self):
        return len(self.dataset)

class GenVQAPredictionDataset(Dataset):
    def __init__(self, tokenizer, dataset_file, img_dir, batch_size=32):
        with open(dataset_file, 'r', encoding='utf-8-sig') as f:
            self.dataset = pd.read_csv(f)

        self.dataset['long_answer'] = self.dataset['long_answer'].apply(literal_eval)
        
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, str(self.dataset.iloc[idx]['image']))
        image = Image.open(img_path, mode='r')
        preprocessed_image = self.preprocessor(image, return_tensors='pt')['pixel_values'].squeeze()
            
        q = self.dataset.iloc[idx]['question']
        tokenized_sentence = self.tokenizer(q)
        input_ids = tokenized_sentence['input_ids']
        attention_mask = tokenized_sentence['attention_mask']

        long_answer = self.dataset.iloc[idx]['long_answer']
        image_id = self.dataset.iloc[idx]['image_id']
        image_path = self.dataset.iloc[idx]['image']
        cat_id = self.dataset.iloc[idx]['category_id']
        formal = self.dataset.iloc[idx]['formal']

        return input_ids, attention_mask, preprocessed_image, long_answer, image_id, image_path, cat_id, formal
        

    
    def __len__(self):
        return len(self.dataset)
    
    
def pad_batched_train_sequence(batch):
    
    input_ids = [torch.tensor(item[0]) for item in batch]
    attention_mask =  [torch.tensor(item[1]) for item in batch]
    images = [item[2] for item in batch]
    
    input_ids = pad_sequence(input_ids, padding_value= 1, batch_first=True).cuda()
    attention_mask = pad_sequence(attention_mask, padding_value=1, batch_first=True).cuda()
    label_tokenized = None

    
    if batch[0][3]:
        label_tokenized = [torch.tensor(item[3]) for item in batch]
        label_tokenized = pad_sequence(label_tokenized, batch_first=False, padding_value=1).cuda()
    # torch.stack(images, dim=0).cuda()
    return input_ids, attention_mask, torch.stack(images, dim=0).cuda(), label_tokenized


def pad_batched_evaluate_sequence(batch):
    
    input_ids = [torch.tensor(item[0]) for item in batch]
    attention_mask =  [torch.tensor(item[1]) for item in batch]
    images = [item[2] for item in batch]
    
    input_ids = pad_sequence(input_ids, padding_value= 1, batch_first=True).cuda()
    attention_mask = pad_sequence(attention_mask, padding_value=1, batch_first=True).cuda()
    label_tokenized = [item[3] for item in batch]

    return input_ids, attention_mask, torch.stack(images, dim=0).cuda(), label_tokenized


def pad_batched_predict_sequence(batch):
    
    input_ids = [torch.tensor(item[0]) for item in batch]
    attention_mask =  [torch.tensor(item[1]) for item in batch]
    
    input_ids = pad_sequence(input_ids, padding_value= 1, batch_first=True).cuda()
    attention_mask = pad_sequence(attention_mask, padding_value=1, batch_first=True).cuda()
    
    images = [item[2] for item in batch]
    long_answer = [item[3] for item in batch]
    image_id = [item[4] for item in batch]
    image_path = [item[5] for item in batch]
    cat_id = [item[6] for item in batch]
    formal= [item[7] for item in batch]

    return input_ids, attention_mask, torch.stack(images, dim=0).cuda(), long_answer, image_id, image_path, cat_id, formal