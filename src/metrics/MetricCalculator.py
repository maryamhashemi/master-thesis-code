import torch
import evaluate
from src.metrics.EmbeddingBase.AverageScore import AverageScore

#!pip install evaluate
#!pip install rouge_score
#!pip install bert_score

class MetricCalculator():
    def __init__(self, embedding_layer, tokenizer) -> None:

        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.METRICS = ["average_score", "bleu", "rougeL", "meteor", "bertscore"]
        self.accumelated_instances = []
        
        self.BLEU = evaluate.load('bleu')
        self.ROUGE = evaluate.load('rouge')
        self.METEOR = evaluate.load('meteor')
        self.BERTSCORE = evaluate.load("bertscore")

    def add_batch(self, preds, references):
        metrics = {"average_score" : AverageScore()}
        
        preds_ids = [self.tokenizer(pred)['input_ids'] for pred in preds]
        ref_ids = [[self.tokenizer(r)['input_ids'] for r in ref] for ref in references]
        
        preds_emb = [self.embedding_layer(torch.tensor(p, dtype=torch.int).cuda()) for p in preds_ids]
        ref_emb = [[self.embedding_layer(torch.tensor(r, dtype=torch.int).cuda()) for r in ref] for ref in ref_ids]
        
        result = {key: metrics[key].compute(preds_emb, ref_emb) for key in metrics}
            
        self.BLEU.add_batch(predictions=preds, references=references)
        self.ROUGE.add_batch(predictions=preds, references=references)
        self.METEOR.add_batch(predictions=preds, references=references)
        self.BERTSCORE.add_batch(predictions=preds, references=references)
        
        self.accumelated_instances.append(result)
        return 

    def compute(self):
        result = {self.BLEU.name: self.BLEU.compute(),
                  self.ROUGE.name:self.ROUGE.compute(tokenizer=lambda x: x.split()),
                  self.METEOR.name: self.METEOR.compute(),
                  self.BERTSCORE.name: self.BERTSCORE.compute(lang="fa")}

        avg_bert_keys = ['precision', 'recall', 'f1']

        for key in avg_bert_keys:
            result[self.BERTSCORE.name][key] = sum(result[self.BERTSCORE.name][key]) / len(result[self.BERTSCORE.name][key])

        average_scores = [item['average_score'].mean for item in self.accumelated_instances]
        result['average_score'] = torch.mean(torch.stack(average_scores)).cpu().tolist()

        return result