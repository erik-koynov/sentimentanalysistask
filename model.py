import torch.nn as nn
from transformers import BertModel
from typing import List
from aggregation_module import AggregationModule
import torch

class Model(nn.Module):
    def __init__(self, companies_list: List[str], n_classes = 4):
        super().__init__()
        self.base = BertModel.from_pretrained("bert-base-uncased")
        self.n_classes = n_classes
        self.module_dict = nn.ModuleDict([
            ["general", nn.Linear(self.base.pooler.dense.out_features, self.n_classes)],
            *[[company, nn.Linear(self.base.pooler.dense.out_features, self.n_classes)] for company in companies_list]
        ])
        self.aggregation_dict = nn.ModuleDict([
            [company, AggregationModule(self.n_classes)] for company in companies_list
        ])

    def forward(self, texts: torch.LongTensor, company_labels: List[str] = None):
        if company_labels is not None and len(company_labels)!=texts.shape[0]:
            raise ValueError(f"company_labels should be the same length as the texts batch dimension.")

        out = self.base(texts, ).pooler_output
        general_out = self.module_dict['general'](out)
        for i, company in enumerate(company_labels):
            try:
                company_out = self.module_dict[company](out[i][None,...])
                general_out = self.aggregation_dict[company](general_out, company_out)
            except KeyError:
                pass
        return general_out