import argparse
import gc
import json
import os
import random
from datetime import datetime
from unittest import result

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.constants import CHECKPOINTS_DIR, HIDDEN_SIZE
from src.data.datasets import GenVQADataset, GenVQAPredictionDataset, pad_batched_train_sequence, pad_batched_evaluate_sequence, pad_batched_predict_sequence
from src.decoders.greedy_decoder import GreedyDecoder
from src.logger import Instance as Logger
from src.metrics.MetricCalculator import MetricCalculator
from src.models import Encoder_AttnRNN, Encoder_RNN, Encoder_Transformer
from src.utils import EarlyStopping
from torch.utils.data.dataloader import DataLoader
from torchmetrics import Accuracy, F1Score, BLEUScore
from tqdm import tqdm


class VQA:
    def __init__(self,
                 mode,
                 train_date,
                 model,
                 decoder_type,
                 train_dset_file = "../pars_vqa/parsvqa_train.csv",
                 val_dset_file= "../pars_vqa/parsvqa_val.csv",
                 test_dset_file= "../pars_vqa/parsvqa_test.csv",
                 img_dir = "../img_data",
                 use_cuda=True,
                 batch_size=32,
                 epochs=50,
                 lr=0.005,
                 log_every=1,
                 save_every=50, 
                 max_sequence_length=50, 
                 optimizer = 'adam',
                 max_norm = 5):
        
        self.mode = mode
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_every = log_every
        self.train_date_time = train_date
        self.save_every = save_every
        self.decoder_type = decoder_type
        self.max_sequence_length = max_sequence_length
        self.max_norm = max_norm
        self.train_dset_file = train_dset_file
        self.val_dset_file = val_dset_file
        self.test_dset_file = test_dset_file
        self.img_dir = img_dir

        if(use_cuda):
            self.model = self.model.cuda()
            
        self.pad_idx = model.tokenizer.pad_token_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        
        if optimizer == 'adam':
            self.optim = torch.optim.Adam(list(self.model.parameters()), lr=lr)
        elif optimizer =='sgd':
            self.optim = torch.optim.SGD(list(self.model.parameters()), lr=lr)
        elif optimizer =='adamw':
            self.optim = torch.optim.AdamW(list(self.model.parameters()), lr=lr)
            

        self.early_stopping = EarlyStopping(patience=3, verbose=True)
        
        self.f1_score = F1Score(num_classes=self.model.tokenizer.vocab_size, ignore_index=self.pad_idx, top_k=1, mdmc_average='samplewise')
        self.accuracy = Accuracy(num_classes=self.model.tokenizer.vocab_size, ignore_index=self.pad_idx, top_k=1, mdmc_average='samplewise')
        
        self.save_dir = os.path.join(CHECKPOINTS_DIR, f'{self.model.name}_{str(self.train_date_time)}')
        if not(os.path.exists(self.save_dir)):
            os.makedirs(self.save_dir, exist_ok=True)
        
    def train(self):
        train_dset = GenVQADataset(tokenizer=self.model.tokenizer, 
                                   dataset_file=self.train_dset_file, 
                                   img_dir = self.img_dir, 
                                   explode=True, 
                                   batch_size=args.batch_size)
        
        train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=True, drop_last=True, collate_fn=pad_batched_train_sequence)
        
        best_bleu_score = 0
        
        for epoch in range(self.epochs):
            running_loss, running_accuracy, running_f1 = self.__training_loop(train_loader, epoch)

            if epoch % self.log_every == self.log_every - 1:                             
                bleu_score = self.__validation_loop()
                
                Logger.log(f"{self.mode}_{self.model.name}_{self.train_date_time}", f"Training epoch {epoch}: Train loss {running_loss:.3f}."
                            + f" Train accuracy {running_accuracy:.3f}. Train F1-Score: {running_f1}. Validation Bleu Score: {bleu_score:.3f}.")
                print(f"Train F1 Score: {running_f1}, Validation Bleu Score: {bleu_score:.3f}")
                
                best_bleu_score = self.__save_best_model(best_bleu_score, bleu_score)
            
            if(epoch % self.save_every == self.save_every - 1):
                self.model.save(self.save_dir, epoch)
    
            
            self.early_stopping(bleu_score)
            if self.early_stopping.early_stop:
                Logger.log(f"{self.mode}_{self.model.name}_{self.train_date_time}", "Early stopping")
                print("Early stopping")
                break
            
            torch.cuda.empty_cache()
            gc.collect()

    def __save_best_model(self, best_bleu_score, bleu_score):
        if(bleu_score > best_bleu_score):
            self.model.save(self.save_dir, "BEST")
            best_bleu_score = bleu_score
                    
            Logger.log(f"{self.mode}_{self.model.name}_{self.train_date_time}", "Save BEST model.")
            print('Save BEST model.')
            
        return best_bleu_score

    def __training_loop(self, train_loader, epoch):
        running_loss = running_accuracy = running_f1 = 0
        
        self.model.train()
        
        pbar = tqdm(train_loader, total=len(train_loader))
        for i, (input_ids, masks, images, target) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch}")
            loss, batch_acc, batch_f1 = self.__step(input_ids, masks, images, target, val=False)  
                
            running_loss += loss.item()
            running_accuracy += batch_acc.item()
            running_f1 += batch_f1
            pbar.set_postfix(loss=running_loss/(i+1), accuracy=running_accuracy/(i+1))
        
        total_data_iterated = self.log_every * len(train_loader)
        running_loss /= total_data_iterated
        running_accuracy /= total_data_iterated
        running_f1 /= total_data_iterated
                
        return running_loss, running_accuracy, running_f1

    @torch.no_grad()
    def __validation_loop(self):
        val_dset = GenVQADataset(self.model.tokenizer, dataset_file=self.val_dset_file, img_dir = self.img_dir, explode=False, batch_size=4)
        loader = DataLoader(val_dset, batch_size=4, shuffle=False, drop_last=True, collate_fn=pad_batched_evaluate_sequence)
        
        decoder = GreedyDecoder(self.model.tokenizer)
        
        bleuScore = BLEUScore()
        
        self.model.eval()
        bleu_score = 0
                
        pbar = tqdm(loader, total=len(loader))
        for j, (input_ids, masks, images, target) in enumerate(pbar):
            logits = self.__get_model_logits(input_ids, masks, images, target, val=True)

            preds_tokenized = decoder.decode_from_logits(logits)
            pred_sentences_decoded = decoder.batch_decode(preds_tokenized.permute(1, 0))
                    
            bleu_score += bleuScore(pred_sentences_decoded, target).item()
            pbar.set_postfix(bleu_score=bleu_score/(j+1))
            
        bleu_score /= len(loader)
        
        return bleu_score
        
    def __step(self, input_ids, masks, images, target, val=False):
        
        logits = self.__get_model_logits(input_ids, masks, images, target, val)
        
        # logits shape: (L, N, target_vocab_size)

        if self.decoder_type == 'transformer':
            target = target[1:,:]
        
        if val:
            target = F.pad(input=target, pad=(0, 0, 0, self.max_sequence_length - target.shape[0]), mode='constant', value=self.pad_idx)
            
        loss = self.criterion(logits.permute(1, 2, 0), target.permute(1,0))

        if not(val):
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
            self.optim.step()

        f1_score = self.f1_score(logits.permute(1,2,0), target.permute(1,0))
        acc = self.accuracy(logits.permute(1,2,0), target.permute(1,0))

        return loss, acc, f1_score

    def __get_model_logits(self, input_ids, masks, images, target, val):
        teacher_force_ratio = 0 if val else 0.5
        answer_tokenized = None if val else target      
        return self.model(input_ids, masks, images, answer_tokenized, teacher_force_ratio, self.max_sequence_length)

    @torch.no_grad()
    def evaluate_metrics(self, key):
        self.model.eval()
        
        if key == 'VAL':
            val_dset = GenVQADataset(self.model.tokenizer, dataset_file=self.val_dset_file, img_dir = self.img_dir, explode=False, batch_size=self.batch_size)
            loader = DataLoader(val_dset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=pad_batched_evaluate_sequence)
        elif key =='TEST':
            test_dset = GenVQADataset(self.model.tokenizer, dataset_file=self.test_dset_file, img_dir = self.img_dir, explode=False, batch_size=self.batch_size)
            loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=pad_batched_evaluate_sequence)

        
        metric_calculator = MetricCalculator(self.model.embedding_layer, self.model.tokenizer)
        decoder = GreedyDecoder(self.model.tokenizer)
            
        pbar = tqdm(loader, total=len(loader))
        for input_ids, masks, images, target in pbar:
            logits = self.__get_model_logits(input_ids, masks, images, target, val=True)

            preds_tokenized = decoder.decode_from_logits(logits)
            pred_sentences_decoded = decoder.batch_decode(preds_tokenized.permute(1, 0))
            
            metric_calculator.add_batch(pred_sentences_decoded, target)
        
        metrics = metric_calculator.compute()
        
        gc.collect()
        
        with open(os.path.join(self.save_dir, f"evaluation_{key}.json"), 'w') as fp:
            json.dump(metrics, fp)
    
    @torch.no_grad()
    def evaluate_metrics_per_category(self):
        self.model.eval()
        
        result={}
        
        for cat_id in range(1,12):
            test_dset = GenVQADataset(self.model.tokenizer, self.test_dset_file, self.img_dir, False, self.batch_size, cat_id)
            loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, drop_last=False, collate_fn=pad_batched_evaluate_sequence)

            metric_calculator = MetricCalculator(self.model.embedding_layer, self.model.tokenizer)
            decoder = GreedyDecoder(self.model.tokenizer)
            
            pbar = tqdm(loader, total=len(loader))
            for input_ids, masks, images, target in pbar:
                logits = self.__get_model_logits(input_ids, masks, images, target, val=True)

                preds_tokenized = decoder.decode_from_logits(logits)
                pred_sentences_decoded = decoder.batch_decode(preds_tokenized.permute(1, 0))
                
                metric_calculator.add_batch(pred_sentences_decoded, target)
            
            metrics = metric_calculator.compute()
            result[cat_id] = metrics
            gc.collect()
        
        with open(os.path.join(self.save_dir, "evaluation_test_per_category.json"), 'w') as fp:
            json.dump(result, fp)
            
    def load_model(self, model_path):
        if model_path == 'BEST':
            model_path = os.path.join(self.save_dir, f"{self.model.name}.{model_path}.torch")
        
        if not (os.path.exists(model_path)):
            Logger.log(f"{self.mode}_{self.model.name}_{self.train_date_time}", f"Couldn't load model from {model_path} ")
            return

        state_dict = torch.load(model_path)
        self.model.load_state_dict(state_dict)
    
    @torch.no_grad()
    def predict(self, model_path, key):
        if key == 'VAL':
            val_dset = GenVQAPredictionDataset(self.model.tokenizer, dataset_file=self.val_dset_file, img_dir = self.img_dir, batch_size=self.batch_size)
            loader = DataLoader(val_dset, batch_size=self.batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_predict_sequence)
        elif key =='TEST':
            test_dset = GenVQAPredictionDataset(self.model.tokenizer, dataset_file=self.test_dset_file, img_dir = self.img_dir, batch_size=self.batch_size)
            loader = DataLoader(test_dset, batch_size=self.batch_size, shuffle=False, drop_last=True, collate_fn=pad_batched_predict_sequence)
            
        self.model.eval()
        decoder = GreedyDecoder(self.model.tokenizer)
        questions, pred_sentences, ref_sentences, image_ids, image_paths, cat_ids, formals = [], [], [], [], [], [], []
        
        pbar = tqdm(loader, total=len(loader))
        
        for input_ids, masks, images, target, image_id, image_path, cat_id, formal in pbar:
            logits = self.__get_model_logits(input_ids, masks, images, target, val=True)
            
            preds_tokenized = decoder.decode_from_logits(logits)
            
            questions_decoded= decoder.batch_decode(input_ids)
            pred_sentences_decoded= decoder.batch_decode(preds_tokenized.permute(1, 0))
            
            questions.extend(questions_decoded)
            pred_sentences.extend(pred_sentences_decoded)
            ref_sentences.extend(target)
            image_ids.extend(image_id)
            image_paths.extend(image_path)
            cat_ids.extend(cat_id)
            formals.extend(formal)
            
            
        model_predictions = [{"image_id": int(image_id) ,
                              "image":image_path,
                              "question":question,
                              "ref answer": ref_answer,
                              "pred answer":pred_answer,
                              'category_id': int(cat_id),
                              'formal':int(formal)} 
                             for image_id, image_path, question, ref_answer, pred_answer, cat_id, formal in 
                             zip(image_ids, image_paths, questions, ref_sentences, pred_sentences, cat_ids, formals)]
              
        with open(os.path.join(os.path.split(model_path)[0], f"model_prediction_{key}.json"), 'w', encoding='utf-8-sig') as fp:
            json.dump(model_predictions, fp)
            
        gc.collect()
                       

def parse_args():
    parser = argparse.ArgumentParser()
    # specify mode, options: train, evaluate, predict:
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--optimizer", default="adam", type=str)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    
    # specify model_path to load for prediction
    parser.add_argument("--model_path", default='', type=str)
    
    # specify seed for reproducing
    parser.add_argument("--seed", default=8956, type=int)
    
    # specify encoder type, options: clip-fa 
    parser.add_argument("--encoder_type", default="clip-fa", type=str)
    
    # specify decoder type, options: rnn, attn-rnn, transformer
    parser.add_argument("--decoder_type", default="rnn", type=str)
    
    # RNN specifications
    parser.add_argument("--rnn_type", default="lstm", type=str) #options: lstm, gru
    parser.add_argument("--num_rnn_layers", default=1, type=int)
    parser.add_argument("--bidirectional", default=False, action="store_true")
    
    # Attention RNN specifications
    parser.add_argument("--attn_type", default="bahdanau", type=str) #options: bahdanau, luong
    # use only when attention type is luong
    parser.add_argument("--attn_method", default="dot", type=str) #options: dot, general, concat
    
    # Transformer specifications
    parser.add_argument("--nheads", default=12, type=int)
    parser.add_argument("--num_transformer_layers", default=6, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    model = None
    
    if (args.decoder_type.lower() == 'rnn'):
        model = Encoder_RNN.Encoder_RNN(encoder_type=args.encoder_type,
                                        rnn_type=args.rnn_type, 
                                        num_layers=args.num_rnn_layers, 
                                        bidirectional=args.bidirectional)
    
    elif(args.decoder_type.lower() == 'attn-rnn'):
        model = Encoder_AttnRNN.Encoder_AttnRNN(encoder_type=args.encoder_type,
                                                rnn_type=args.rnn_type,
                                                attn_type = args.attn_type,
                                                attn_method=args.attn_method)
        
    elif (args.decoder_type.lower() == 'transformer'):
        model = Encoder_Transformer.Encoder_Transformer(encoder_type=args.encoder_type,
                                                        nheads=args.nheads,
                                                        decoder_layers=args.num_transformer_layers,
                                                        hidden_size=HIDDEN_SIZE)
    
    if model:
        vqa = VQA(
            mode=args.mode, 
            train_date= datetime.now(), 
            model=model, 
            decoder_type=args.decoder_type, 
            optimizer=args.optimizer, 
            lr= args.lr,
            batch_size=args.batch_size
        )
        
        if args.mode == 'train':
            Logger.log(f"{args.mode}_{vqa.model.name}_{vqa.train_date_time}", f"{args.optimizer}-{args.lr}-{args.batch_size}-{model.name}")
            vqa.train()
            vqa.load_model("BEST")
            vqa.evaluate_metrics("VAL")
            vqa.evaluate_metrics("TEST")
                
        elif args.mode =='predict':
            Logger.log(f"{args.mode}_{vqa.model.name}_{vqa.train_date_time}", f"{args.model_path}")
            vqa.load_model(args.model_path)
            vqa.predict(args.model_path, "TEST")
        
        elif args.mode == 'evaluate':
            Logger.log(f"{args.mode}_{vqa.model.name}_{vqa.train_date_time}", f"{args.model_path}")
            vqa.load_model(args.model_path)
            vqa.evaluate_metrics("VAL")
            vqa.evaluate_metrics("TEST")
            vqa.evaluate_metrics_per_category()
