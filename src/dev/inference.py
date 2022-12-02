import config

import torch

import pickle
from torchaudio.models.decoder import ctc_decoder

class GreedyCTCDecoder():
    def __init__(
        self,
        decoder_path: str = str(config.DICTIONARY_PATH/"decode_dict.pkl"),
        blank_idx: int = 0,
    ):
        self.decoder_path = decoder_path
        self.blank_idx = blank_idx
        with open(self.decoder_path, "rb") as f:
            self.decode_dict = pickle.load(f)
        self.blank_token = self.decode_dict[self.blank_idx]
        
    def decode(self, emissions: torch.tensor):
        ## Emission: (bs, seq_length, no_tokens)
        preds_id_full = torch.argmax(emissions, dim = 2) # (bs, seq_length)
        
        preds_ids = []
        preds_full = []
        preds = []
        
        for pred_id in preds_id_full:
            pred_collapsed = torch.unique_consecutive(pred_id).detach().cpu().numpy()
            preds_ids.append(pred_collapsed)
            pred_sequence_full = "".join([self.decode_dict[idx.item()] for idx in pred_id])
            preds_full.append(pred_sequence_full)
            pred_sequence = "".join([self.decode_dict[idx] for idx in pred_collapsed if self.decode_dict[idx] != self.blank_token])
            preds.append(pred_sequence)
        return preds_ids, preds_full, preds
    
class BeamSearchCTCDecoder():
    def __init__(
        self,
        decoder_path: str = str(config.DICTIONARY_PATH/"decode_dict.pkl"),
        beam_size: int = 50,
        blank_idx: int = 0,
    ):
        self.decoder_path = decoder_path
        self.blank_idx = blank_idx
        with open(self.decoder_path, "rb") as f:
            self.decode_dict = pickle.load(f)
        
        self.blank_token = self.decode_dict[self.blank_idx]
        # self.tokens = [self.decode_dict[idx] for idx in range(len(self.decode_dict))] 
        self.tokens = [self.decode_dict[idx] for idx in range(len(self.decode_dict))] + ["|"]
        self.decoder = ctc_decoder(lexicon = None, tokens = self.tokens, beam_size = beam_size, blank_token = self.blank_token)
        # self.decode_dict[len(self.tokens)-1] = "|"
        
    def decode(self, emissions: torch.tensor):
        ## Emission = (bs, seq_len, token_number)
        _, seq_len, _ = emissions.shape
        hypotheses = self.decoder(emissions)
        
        preds_ids = []
        preds_full = []
        preds = []
        
        for hypothesis in hypotheses:
            best_beam = hypothesis[0]
            pred_collapsed = best_beam.tokens.detach().cpu().numpy()
            pred_collapsed = pred_collapsed[1:(len(pred_collapsed)-1)]
            preds_ids.append(pred_collapsed)
            full_token_list = [self.blank_token] * seq_len
            for idx in range(1, len(best_beam.timesteps)-1):
                full_token_list[best_beam.timesteps[idx].item()-1] = self.decode_dict[best_beam.tokens[idx].item()]
            preds_full.append("".join(full_token_list))
            pred_sequence = "".join([self.decode_dict[idx] for idx in pred_collapsed])
            preds.append(pred_sequence)
            
        return preds_ids, preds_full, preds
    
if __name__ == "__main__":
    # greedy_decoder = GreedyCTCDecoder()
    # probs = torch.rand(5, 75, len(greedy_decoder.decode_dict))
    # log_probs = torch.softmax(probs, dim = 2)
    # print(greedy_decoder.decode(log_probs))
    
    beam_decoder = BeamSearchCTCDecoder()
    probs = torch.rand(5, 75, len(beam_decoder.decode_dict))
    log_probs = torch.softmax(probs, dim = 2)
    print(beam_decoder.decode(log_probs))