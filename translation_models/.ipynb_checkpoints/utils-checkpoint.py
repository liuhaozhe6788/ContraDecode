import torch
from typing import List
from transformers.generation_logits_process import  LogitsProcessorList, LogitsProcessor, ForcedBOSTokenLogitsProcessor


def batch(input: List, batch_size: int):
    l = len(input)
    for ndx in range(0, l, batch_size):
        yield input[ndx:min(ndx + batch_size, l)]

def zero_out_max(x):
    max_num, max_index = x.max(dim=0)
    x[max_index] = 0
    return x


class EnsembleLogitsProcessor(LogitsProcessor):

    def __init__(self, num_beams: int, source_weights: List[float] = None, preserve_bos_token: bool = False):
        self.num_beams = num_beams
        self.source_weights = source_weights
        self.preserve_bos_token = preserve_bos_token

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        cur_len = input_ids.shape[-1]
        if self.preserve_bos_token and cur_len <= 1:
            return scores

        # scores = F.softmax(scores, dim=-1)
        scores = torch.exp(scores)

        batch_size = int(input_ids.size(0) / self.num_beams)
        if self.source_weights is not None:
            assert len(self.source_weights) == batch_size
            source_weights = torch.Tensor(self.source_weights).to(scores.device)
        else:
            source_weights = 1/(batch_size-1) * torch.ones((batch_size,), device=scores.device)
        for i in range(self.num_beams):
            beam_indices = self.num_beams * torch.arange(batch_size, device=scores.device, dtype=torch.long) + i
            cands = scores[beam_indices]
            mean_scores = torch.log((source_weights.unsqueeze(-1).expand(-1, scores.size(-1)) * cands).sum(dim=0))
            for j in beam_indices:
                scores[j] = mean_scores

        if torch.isnan(scores).any():
            scores = torch.nan_to_num(scores, nan=float('-inf'))

        return scores