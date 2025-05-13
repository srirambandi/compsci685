import torch
import torch.nn.functional as F
import pdb

class SCINComputer():
    '''
    This class computes ||orth()|| for all spans in given sentences.
    '''
    def __init__(self, orth_bidir):
        self.orth_bidir = orth_bidir

    def compute_required_spans(self, parse, word_boundaries):
        # helper function to comptue which spans we need SCIN for.
        # every span in parse greater than equal to 2 in size will be split.
        required_spans = []
        if parse is None:
            for st in range(len(word_boundaries)):
                for en in range(st, len(word_boundaries)):
                    required_spans.append((st,en))
            return required_spans

        for span in parse:
            [st, en] = [int(_) for _ in span.split(" ")]
            en_real = en - 1 # second index is off by 1
            if (en_real - st > 1):
                for idx in range(st+1,en):
                    if word_boundaries[idx]:
                        required_spans.append((st,idx-1))
                        required_spans.append((idx,en_real))
                        if self.orth_bidir:
                            required_spans.append((idx,idx))
                            if en < len(word_boundaries):
                                required_spans.append((en,en))

        return list(set(required_spans))

    def get_all_orthogonal_scores(self, hidden_states):
        # Get norm of orthogonals between all pairs of hidden states using vectorization.
        norm_all_vector_idxs = F.normalize(hidden_states, dim=-1)
        # norm_all_vector_idxs = hidden_states

        orthogonals = hidden_states - torch.sum(hidden_states * norm_all_vector_idxs.unsqueeze(dim=1), dim=-1).unsqueeze(-1) * norm_all_vector_idxs.unsqueeze(dim=1)
        # pdb.set_trace()

        return torch.norm(orthogonals, dim=-1)
    
    def get_batched_orthogonal_scores(self, hidden_states, st, ens):
        # Get norm of orthogonals between the hidden state at index st, and all hidden states ending at ens.
        context = F.normalize(hidden_states[st].unsqueeze(0), dim=-1)
        # context = hidden_states[st].unsqueeze(0)
        span_vectors = hidden_states[ens]
        components_along = span_vectors@context.T
        # pdb.set_trace()

        return torch.norm(span_vectors - (components_along @ context), dim=-1)

    def build_chart(self, hidden_states, word_boundaries, parses):
        '''
        Compute the ||orth()|| for all required spans in the given sentences.
        hidden_states: For a batch of sentences, the appropriate hidden states from the model after a forward pass.
        word_boundaries: A boolean array with 1s at all indices where words start in tokenized input sentences.
        parses: The silver parses, if computation of  the TreeReg loss is required.
        '''
        scores = [{} for _ in word_boundaries]
        # hidden_states = F.normalize(hidden_states, dim=-1)

        for idx, curr_word_boundaries in enumerate(word_boundaries):
            num_tokens = len(curr_word_boundaries)
            curr_parse = None if parses is None else parses[idx]
            required_spans = self.compute_required_spans(curr_parse, word_boundaries[idx])
            # if most of the span SCINs require computation, do computation for all spans at once
            if len(required_spans) > num_tokens*num_tokens//4:
                orthogonal_magnitudes = self.get_all_orthogonal_scores(hidden_states[idx].squeeze(0))
                for (st, en) in required_spans:
                    if (st == 0):
                        scores[idx][(st,en)] = 0
                    else:
                        scores[idx][(st,en)] = orthogonal_magnitudes[st-1][en]
                scores[idx][(num_tokens, num_tokens)] = 0
            else:
                # doing all computation is wasteful, batch the required ones instead
                required_spans.sort()
                batch_end_points = []
                curr_start = 0
                for (st, en) in required_spans:
                    if st == 0:
                        scores[idx][(st,en)] = 0
                        continue

                    if st == curr_start:
                        batch_end_points.append(en)
                    else:
                        # compute the batched orthogonals
                        if curr_start != 0:
                            orthogonal_magnitudes = self.get_batched_orthogonal_scores(hidden_states[idx].squeeze(0), curr_start - 1, batch_end_points)
                            for iidx, end_point in enumerate(batch_end_points):
                                scores[idx][(curr_start,end_point)] = orthogonal_magnitudes[iidx]
                        curr_start = st
                        batch_end_points = [en]
                    
                if (len(batch_end_points) > 0):
                    orthogonal_magnitudes = self.get_batched_orthogonal_scores(hidden_states[idx].squeeze(0), curr_start - 1, batch_end_points)
                    for iidx, end_point in enumerate(batch_end_points):
                        scores[idx][(curr_start,end_point)] = orthogonal_magnitudes[iidx]
                scores[idx][(num_tokens, num_tokens)] = 0

        return scores
                            