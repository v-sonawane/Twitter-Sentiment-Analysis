import config
import transformers
import torch.nn as nn


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.bert_drop = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        sequence_output, pulled_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(pulled_output)
        output = self.out(bo)
        logits=self.l0(sequence_output)
        start_logits,end_logits=logits.split(1,dim=-1)
        start_logits=start_logits.squeeze(1)
        end_logits=end_logits.squeeze(1)


        return start_logits,end_logits