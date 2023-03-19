from torch.utils.data import Dataset
import pandas as pd
from typing import List, Tuple, Union
import torch
import numpy as np
from transformers import BertTokenizer
from pathlib import Path

class Dataset4Pandas(Dataset):
    def __init__(self,
                dataframe: pd.DataFrame,
                text_column: str,
                label_column: str,
                company_column: str,
                irrelevant_label_encoding = 3
                ):
        self.dataframe = dataframe.dropna().reset_index(drop=True)
        self.text_column = text_column+"_tokenized"
        self.label_column = label_column
        self.company_column = company_column
        self.companies = self.dataframe[self.company_column].unique()
        self.irrelevant_label_encoding = irrelevant_label_encoding
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.dataframe.loc[:, self.text_column] = self.dataframe[text_column].apply(self._apply_tokenization)

    @classmethod
    def from_csv(cls, csv_path: Union[str, Path],
                 text_column: str,
                 label_column: str,
                 company_column: str):
        dataframe = pd.read_csv(csv_path)
        return cls(dataframe, text_column, label_column, company_column)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx)->Tuple[torch.Tensor, torch.Tensor, str]:
        item = self.dataframe.iloc[idx].copy()
        label = item[self.label_column]
        if label == self.irrelevant_label_encoding: # irrelevants can be randomly sampled
            company = np.random.choice(self.companies)
        else:
            company = item[self.company_column]

        return torch.LongTensor(item[self.text_column]),\
               torch.LongTensor([label]), \
               company

    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, str]])->Tuple[torch.Tensor, torch.Tensor, Tuple[str]]:
        batch_members = list(zip(*batch))
        batch_members[0] = torch.cat([tensor[None,:] for tensor in Dataset4Pandas.pad_sentence_sequence_to_max(list(batch_members[0]))])
        batch_members[1] = torch.cat(batch_members[1])
        return tuple(batch_members)

    @staticmethod
    def pad_sentence_sequence_to_max(batch: List[torch.Tensor], pad_digit=0) -> List[torch.Tensor]:
        """
        Collate a batch of sequences of sequence embeddings such that the sequences are of the
        same length -> e.g. one sequence might be of 2 sequence embeddings, the other of 3 ->
        pad the shorter to the length of the longest sequence in the batch.
        :param batch: a list of tensors (the sequences of sequence embeddings)
        :return: list of the thus padded sequences
        """
        seq_lengths = np.array([len(i) for i in batch])

        # select the items in the batch that have smaller seq length than the max seq length in the batch
        to_pad = np.where(seq_lengths < max(seq_lengths))[0]

        for i in to_pad:
            needed_count = max(seq_lengths) - len(batch[i])
            batch[i] = torch.cat((batch[i], torch.full((needed_count.item(),), pad_digit, dtype=batch[i].dtype)))

        return batch

    def _apply_tokenization(self, element: str):

        return self.tokenizer.encode_plus(element,
                     add_special_tokens=True,
                     max_length=510,
                     truncation=True,
                     truncation_strategy='longest_first',
                     return_attention_mask=False,
                     return_token_type_ids=False)["input_ids"]