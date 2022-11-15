import os

import torch
import torch.nn.functional as F
from src.utils import PositionalEncoder
from torch import nn
from transformers import (AutoTokenizer, CLIPFeatureExtractor, CLIPVisionModel,
                          RobertaModel)


class Encoder_Transformer(nn.Module):
    def __init__(self, encoder_type, nheads, decoder_layers, hidden_size, freeze_encoder=True):
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'clip-fa':
            self.vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
            self.text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
            self.preprocessor = CLIPFeatureExtractor.from_pretrained('SajjadAyoubi/clip-fa-vision')
            self.tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')

            
        #freeze encoder
        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            
            for p in self.text_encoder.parameters():
                p.requires_grad = False

        
        transformer_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nheads)
        self.Decoder = nn.TransformerDecoder(transformer_layer, num_layers=decoder_layers)

        self.pe = PositionalEncoder(hidden_size, dropout=0.1,max_len=200)
        
        self.embedding_layer = self.text_encoder.embeddings.word_embeddings
        self.output_size = self.tokenizer.vocab_size
        self.decoder_layers = decoder_layers
        self.nheads = nheads
        self.hidden_size = hidden_size
        self.PADDING_VALUE = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.bos_token_id
        self.SEP_TOKEN = self.tokenizer.eos_token_id 
        
        self.Linear = nn.Linear(hidden_size, self.output_size)
        
        self.name = f"{encoder_type}_{nheads}heads_{decoder_layers}_transformer"
        print(self.name)
    
    def forward(self, input_ids, attention_mask, image, answer_tokenized=None, teacher_force_ratio=0.5, max_seq_len=50):

        batch_size = input_ids.shape[0]
        max_seq_len = max_seq_len if answer_tokenized is None else answer_tokenized.shape[0]

        # shift right
        answer_tokenized = answer_tokenized[:-1,:] if answer_tokenized is not None else answer_tokenized
        
        # encode question and image
        if self.encoder_type == 'clip-fa':
            text_embedding = self.text_encoder(input_ids, attention_mask).last_hidden_state
            image_embedding = self.vision_encoder(image).last_hidden_state

            # print(text_embedding.shape)
            # print(image_embedding.shape)
            
            # encoder_output shape: (seq_len, batch_size, hidden_size)
            encoder_output  = torch.concat((text_embedding, image_embedding), 1).permute(1,0,2)
            
            # memory masks to consider padding values in source sentence (questions+image)
            memory_key_padding_mask = (input_ids == self.PADDING_VALUE)
            memory_key_padding_mask = F.pad(input=memory_key_padding_mask, pad=(0, encoder_output.shape[0] - memory_key_padding_mask.shape[1]), mode='constant', value=0)
            
        if answer_tokenized is not None:
            tgt_len = answer_tokenized.shape[0]

            answer_embeddings = self.embedding_layer(answer_tokenized)
            # answer embeddings shape: (seq_len, batch_size, embedding_size)

            positions = self.pe(answer_embeddings)
            
            # target masks to consider padding values in target embeddings (answers)
            tgt_key_padding_mask = (answer_tokenized.permute(1, 0) == self.PADDING_VALUE)

            # target attention masks to avoid future tokens in our predictions
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()        
            

            # decode sentence and encoder output to generate answer
            output = self.Decoder(positions, 
                                encoder_output,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask, 
                                memory_key_padding_mask = memory_key_padding_mask)
            #output shape: (tgt_seq_len, batch_size, hidden_size)

            output = self.Linear(output)
            #output shape: (tgt_seq_len, batch_size, vocab_size)
        
        else:

            x = torch.tensor([[self.START_TOKEN] * batch_size]).cuda()
            # x shape: (1, batch_size)
            
            outputs = torch.zeros(max_seq_len, batch_size, self.output_size).cuda()
            
            for i in range(max_seq_len):
                tgt_len = x.shape[0]
                answer_embeddings = self.embedding_layer(x)
                positions = self.pe(answer_embeddings)
                tgt_key_padding_mask = (x.permute(1, 0) == self.PADDING_VALUE)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).cuda()
                output = self.Decoder( 
                                positions, 
                                encoder_output,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask = tgt_key_padding_mask, 
                                memory_key_padding_mask = memory_key_padding_mask) 
                #output shape: (tgt_seq_len, batch_size, hidden_size)
                
                # chose the last word of sequence
                output = output[-1, :, :].unsqueeze(0)
                #output shape (1, batch_size, hidden_size)
                
                output = self.Linear(output)
                #output shape: (1, batch_size, vocab_size)
                
                outputs[i] = output
                
                #consider best guesses in a greeedy form! Better to implement with beam search
                output = torch.argmax(output, dim = -1)
                #output shape: (1, batch_size)

                #concat new generated answer to x.
                x = torch.cat([x, output], dim=0)
                # x shape: (i + 1, batch_size)
            
            output = outputs

        return output

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
