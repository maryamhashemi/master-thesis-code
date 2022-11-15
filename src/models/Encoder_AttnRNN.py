import os
import random

import torch
from transformers import (AutoTokenizer, CLIPFeatureExtractor, CLIPVisionModel,
                          RobertaModel)

from src.models.BahdanauRNN import BahdanauRNN
from src.models.LuongRNN import LuongRNN


class Encoder_AttnRNN(torch.nn.Module):
    def __init__(self, encoder_type='lxmert', rnn_type = "lstm", attn_type='bahdanau', attn_method='dot', prob=0.5, freeze_encoder=True):
        super().__init__()
        self.encoder_type = encoder_type
        
        if encoder_type == 'clip-fa':
            self.vision_encoder = CLIPVisionModel.from_pretrained('SajjadAyoubi/clip-fa-vision')
            self.text_encoder = RobertaModel.from_pretrained('SajjadAyoubi/clip-fa-text')
            self.tokenizer = AutoTokenizer.from_pretrained('SajjadAyoubi/clip-fa-text')
        
        #freeze encoder
        if freeze_encoder:
            for p in self.vision_encoder.parameters():
                p.requires_grad = False
            
            for p in self.text_encoder.parameters():
                p.requires_grad = False
          
        self.embedding_layer = self.text_encoder.embeddings.word_embeddings

        if attn_type=='bahdanau':
            self.rnn = BahdanauRNN(embedding=self.embedding_layer,
                                    rnn_type=rnn_type,  
                                    output_size=self.tokenizer.vocab_size,
                                    prob=prob)
            self.name = f"{encoder_type}_{attn_type}_attn_{rnn_type}"
        elif attn_type =='luong':
            self.rnn = LuongRNN(embedding=self.embedding_layer,
                                rnn_type=rnn_type,
                                attn_method=attn_method,
                                output_size=self.tokenizer.vocab_size,
                                prob=prob)
            self.name = f"{encoder_type}_{attn_type}_attn({attn_method})_{rnn_type}"
        
        self.PADDING_VALUE = self.tokenizer.pad_token_id
        self.START_TOKEN = self.tokenizer.bos_token_id
        self.SEP_TOKEN = self.tokenizer.eos_token_id 
        
        print(self.name)
        
    def forward(self, input_ids, attention_mask, image, answer_tokenized = None, teacher_force_ratio=0.5,  max_sequence_length=50):
        """
            Train phase forward propagation
        """

        batch_size = input_ids.shape[0]
        target_len = max_sequence_length if answer_tokenized is None else answer_tokenized.shape[0]
        target_vocab_size = self.tokenizer.vocab_size
        
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).cuda()

        #encoder
        if self.encoder_type == 'clip-fa':
            text_output = self.text_encoder(input_ids, attention_mask)
            image_output = self.vision_encoder(image)
            
            encoder_states  = torch.concat((text_output.last_hidden_state, image_output.last_hidden_state), 1).permute(1,0,2)
            # encoder_states shape: (seq_len, batch_size, hidden_size)
            
            encoder_output = torch.add(text_output.pooler_output,image_output.pooler_output) / 2
            # encoder_output shape: (batch_size, hidden_size)
            
        h = encoder_output.expand(1, -1, -1)
        # h shape: (1, N, hidden_size)
        
        if self.rnn.rnn_type == 'lstm':
            c = torch.zeros(*h.shape).cuda()
            hidden = (h.contiguous(),c.contiguous())
        elif self.rnn.rnn_type == 'gru':
            hidden = h.contiguous()
            
        # Send <cls> token to decoder
        x =  torch.tensor([self.START_TOKEN]*batch_size).cuda()
        
        for t in range(target_len):
            output, hidden, _ = self.rnn(x, hidden, encoder_states)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the decoder predicted it to be.
            x = answer_tokenized[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

    def save(self, dir_, epoch):
        if not(os.path.exists(dir_)):
            os.makedirs(dir_, exist_ok=True)
        path = os.path.join(dir_, f"{self.name}.{epoch}.torch")
        torch.save(self.state_dict(), path)
