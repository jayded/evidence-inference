from collections import namedtuple

from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_padded_sequence, pad_packed_sequence

PaddedSequence_ = namedtuple('PaddedSequence_', ['data', 'batch_sizes'])


class PaddedSequence(PaddedSequence_):
    """A utility class for padding variable length sequences mean for RNN input

    This class is in the style of PackedSequence from the PyTorch RNN Utils,
    but is somewhat more manual in approach. It provides the ability to generate masks
    for outputs of the same input dimensions.

    The constructor should never be called directly and should only be called via
    the autopad classmethod.

    This originated because PyTorch's PackedPadded originally required sorting by length,
    however this need is no longer met. We looked into removing this class and unfortunately
    it seems that it's still needed. We would very much like to remove this code, but until
    PackedSequences are treated as first-class funny-looking tensors, that appears to be
    impossible.
    """

    def __new__(cls, data: torch.Tensor, batch_sizes: torch.Tensor, batch_first=False):
        self = super(PaddedSequence, cls).__new__(cls, data, batch_sizes)
        self.batch_first = batch_first
        if not batch_first:
            raise ValueError("No models in this file work without batch_first!")
        return self

    @classmethod
    def autopad(cls, data, batch_first: bool=False, padding_value=0) -> 'PaddedSequence':
        padded = pad_sequence(data, batch_first=batch_first, padding_value=padding_value)
        if batch_first:
            batch_lengths = torch.LongTensor([len(x) for x in data])
            if any([x == 0 for x in batch_lengths]):
                raise ValueError("Found a 0 length batch element, this can't possibly be right: {}".format(batch_lengths))
        else:
            # TODO actually test this codepath
            batch_lengths = torch.LongTensor([len(x) for x in data])
        return PaddedSequence(padded, batch_lengths, batch_first)

    def pack_other(self, data: torch.Tensor):
        return pack_padded_sequence(data, self.batch_sizes, batch_first=self.batch_first, enforce_sorted=False)

    @classmethod
    def from_packed_sequence(cls, ps: PackedSequence, batch_first: bool, padding_value=0) -> 'PaddedSequence':
        padded, batch_sizes = pad_packed_sequence(ps, batch_first, padding_value)
        return PaddedSequence(padded, batch_sizes, batch_first)

    def cuda(self) -> 'PaddedSequence':
        return PaddedSequence(self.data.cuda(), self.batch_sizes.cuda(), batch_first=self.batch_first)

    def mask(self, cuda=False) -> torch.Tensor:
        out_tensor = torch.zeros(*self.data.size()[0:2])
        if self.batch_first:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[i, :bl] = torch.ones(bl)
        else:
            for i, bl in enumerate(self.batch_sizes):
                out_tensor[:bl, i] = torch.ones(bl)
        if cuda:
            return out_tensor.cuda()
        else:
            return out_tensor

    def unpad(self, other: torch.Tensor) -> List[torch.Tensor]:
        out = []
        for o, bl in zip(other, self.batch_sizes):
            out.append(o[:bl])
        return out
