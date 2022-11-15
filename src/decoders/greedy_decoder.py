import torch

class GreedyDecoder():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.SEP = tokenizer.eos_token_id
        self.PAD = tokenizer.pad_token_id
    
    def decode_from_logits(self, logits):
        logits = torch.argmax(logits, dim=-1)    
        return logits

    def batch_decode(self, tokens):
        sentences = []
        for i in range(tokens.shape[0]):
            sentence = []
            for j in range(tokens.shape[1]):
                if(tokens[i, j] == self.PAD):
                    continue
                if(tokens[i, j] == self.SEP):
                    break
                sentence.append(tokens[i, j])
            sentences.append(self.tokenizer.decode(sentence, skip_special_tokens=True))
        return sentences