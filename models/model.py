import torch.nn as nn

from transformers import BertModel


class MRC_model(nn.Module):
    def __init__(self, pretrained_model_path):
        super(MRC_model, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.classifier_cls = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                attention_mask=None):
        output = self.bert(input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
        sequence_output, pooled_output = output[0], output[1]
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.chunk(2, dim=-1)
        start_logits = start_logits.squeeze(dim=-1)
        end_logits = end_logits.squeeze(dim=-1)
        cls_logits = self.classifier_cls(pooled_output)

        return start_logits, end_logits, cls_logits

if __name__ == '__main__':
    model = MRC_model("/data/zhoujx/prev_trained_model/rbt3")
