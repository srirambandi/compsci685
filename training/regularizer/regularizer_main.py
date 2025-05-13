import torch
import torch.nn.functional as F
from regularizer.scin_computer import SCINComputer

class TreeRegularizer(torch.nn.Module):
    '''
    This class contains logic for the computation of the TreeReg loss as well as induced parses, given hidden states from a model.
    '''
    def __init__(self, orth_bidir=True):
        '''
        orth_bidir is always set for the proposed formulation of TreeReg.
        A separate SCINComputer object computes the SCIN scores for spans of a given sentence.
        '''
        super().__init__()

        self.orth_bidir = orth_bidir
        self.scin_chart = SCINComputer(orth_bidir)

    def build_chart(self, hidden_states, word_boundaries, parses):
        '''
        Call the SCINComputer to get SCIN scores.
        hidden_states: For a batch of sentences, the appropriate hidden states from the model after a forward pass. 
                        Typically, these will be from a subset of attention heads at an intermediate layer of the model.
        word_boundaries: A boolean array with 1s at all indices where words start in tokenized input sentences.
        parses: The silver parses, if computation of  the TreeReg loss is required. Can make the chart computation faster for long sentences if passed.
        '''
        return self.scin_chart.build_chart(hidden_states, word_boundaries, parses)

    def get_span_score(self, chart, st, k, en):
        '''
        The span_score for span (st,en) split at k.
        span_score(st,k,en) = ||orth(st,k)|| + ||orth(k+1,k+1)|| + ||orth(en+1,en+1)|| + ||orth(k+1,en)||
        '''
        base_score = chart[(st,k)] + chart[(k+1,en)]
        if self.orth_bidir:
            base_score = base_score + chart[(k+1,k+1)]
            base_score = base_score + chart[(en+1,en+1)]
        return base_score

    def gold_recurse(self, chart, word_boundaries, parse, st, en):
        '''
        Compute the TreeReg loss for a tree.
        Returns the loss for current tree, total number of split points in the tree, and total number of spans split at the correct split point according to the silver parse.
        chart: ||orth()|| for all spans in the tree.
        st: start of the current span.
        en: end of the current span.
        '''
        
        if (en - st <= 1):
            return 0, 0, 0
        
        scores = []
        indices = {}
        idx = 0
        for k in range(st, en):
            if word_boundaries[k+1]:
                # this is a possible split
                scores.append(self.get_span_score(chart, st, k, en))
                indices[k] = idx
                idx += 1

        if len(scores) < 2:
            return 0,0,0
        
        scores = torch.stack(scores)
        curr_span = str(st) + " " + str(en + 1)
        best = parse[curr_span] - 1
        best_score = scores[indices[best]]

        if best_score == torch.max(scores):
            is_best = True
        else:
            is_best = False

        s1, n1, r1 = self.gold_recurse(chart, word_boundaries, parse, st, best)
        s2, n2, r2 = self.gold_recurse(chart, word_boundaries, parse, best+1, en)

        span_loss = -F.cross_entropy(scores, torch.tensor(indices[best], dtype = torch.int64).to(scores.device))

        return s1 + s2 + span_loss, n1 + n2 + 1, r1 + r2 + int(is_best)

    def get_score(self, charts, word_boundaries, parses, device):
        '''
        Compute the TreeReg loss for a batch of sentences.
        charts: list ||orth()|| for all spans in the tree corresponding to each sentence (computed from SCINComputer).
        word_boundaries: A boolean array with 1s at all indices where words start in tokenized input sentences for all sentences.
        parses: The silver parses for all sentences.
        device: The device which is to be used for loss computation.
        '''
        scores = []
        tot = 0
        tot_correct = 0
        for idx, chart in enumerate(charts):
            # print(chart)
            end = len(word_boundaries[idx])
            parse = parses[idx]
            word_boundaries_current = word_boundaries[idx]
            score, tot_terms, tot_right = self.gold_recurse(chart, word_boundaries_current, parse, 0, end-1)
            tot_terms = max(tot_terms,1)
            score /= tot_terms
            tot_correct += tot_right
            tot += tot_terms
            if (score == 0):
                score = torch.tensor(0, requires_grad = True, dtype = torch.float).to(device)
            scores.append(score)

        if tot==0:
            return scores, None
        else:
            return scores, tot_correct/tot

    def get_parse(self, input_strs, charts, word_boundaries, separator = " "):
        '''
        Get the induced parse trees according to TreeReg for a batch of input sentences.
        input_strs: Batch of input sentences in string form.
        charts: list ||orth()|| for all spans in the tree corresponding to each sentence (computed from SCINComputer).
        word_boundaries: A boolean array with 1s at all indices where words start in tokenized input sentences for all sentences.
        '''
        def recurse(chart, word_list, word_boundaries_curr, st, en):
            if (st == en):
                return word_list[st], 0
            else:
                one_word = True
                for k in range(st+1,en+1):
                    if word_boundaries_curr[k]:
                        one_word = False
                        break
                if one_word:
                    return "".join(word_list[st:en+1]), 0

                scores = []
                indices = []
                for k in range(st, en):
                    if word_boundaries_curr[k+1]:
                        cand_score = self.get_span_score(chart, st, k, en)
                        scores.append(cand_score)
                        indices.append(k)

                scores = torch.stack(scores)
                best_idx = torch.argmax(scores)
                best = indices[best_idx.item()]

                p1, s1 = recurse(chart, word_list, word_boundaries_curr, st, best)
                p2, s2 = recurse(chart, word_list, word_boundaries_curr, best+1, en)

                return (p1, p2), scores[best_idx] + s1 + s2

        parses = []
        scores = []
        for idx, chart in enumerate(charts):
            # print(chart)
            end = len(word_boundaries[idx])
            word_boundaries_current = word_boundaries[idx]
            curr_str = input_strs[idx]
            word_list = curr_str if len(separator) == 0 else curr_str.split(separator)
            parse, score = recurse(chart, word_list, word_boundaries_current, 0, end-1)
            parses.append(parse)
            scores.append(score)

        return parses, scores