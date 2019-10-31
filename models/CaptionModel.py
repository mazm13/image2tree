import torch
import torch.nn as nn


class CaptionModel(nn.Module):
    def __init__(self):
        super(CaptionModel, self).__init__()

    # implements beam search
    # calls beam_step and returns the final set beams

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'forward')
        if mode in kwargs:
            del kwargs['mode']
        return getattr(self, '_' + mode)(*args, **kwargs)

    def beam_search(self, init_state, init_logporbs, G, L):

        def beam_step(logprobsf, beam_size, beam_logprobs_sum):
            ys, ix = torch.sort(logprobsf, 1, True)
            candidates = []
            cols = min(beam_size, ys.size(0))
            print(ys.size())
            rows = beam_size
            for c in range(cols):
                for q in range(rows):
                    print(q, c)
                    local_logprob = ys[q, c].item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    candidates.append({'c': ix[q, c], 'q': q, 'p': candidate_logprob})
            candidates = sorted(candidates, key=lambda x: -x['p'])
            return candidates

        # # Start beam search
        # opt = kwargs['opt']
        # G = opt.get('global_beam_size', 5)
        # L = opt.get('local_beam_size', 5)

        beam_logprobs_sum_table = torch.zeros(G)
        candidates = beam_step(init_logporbs, G, beam_logprobs_sum_table)
        candidates = candidates[:G]

        print([cdd['c'] for cdd in candidates])
