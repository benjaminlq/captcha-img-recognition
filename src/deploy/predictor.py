from dev import utils
from dev import models
from dev.inference import GreedyCTCDecoder, BeamSearchCTCDecoder
import pickle
import config
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class Predictor:
    def __init__(self, model_path, decoder_path, decode_type = "beam",
                 beam_size = 50, blank_idx = 0):
        self.model_path = model_path
        self.decoder_path = decoder_path

        with open(self.decoder_path, "rb") as f:
            self.decode_dict = pickle.load(f)

        self.model = models.CTC(len(self.decode_dict)-1)
        utils.load_model(self.model, self.model_path)
        self.model.eval()
        self.model.to(config.DEVICE)

        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            ]
        )
        
        if decode_type == "greedy":
            self.decoder = GreedyCTCDecoder(decoder_path, blank_idx = blank_idx)
        else:
            self.decoder = BeamSearchCTCDecoder(decoder_path, beam_size = beam_size, blank_idx = blank_idx)

    def predict(self, input_file):
        input = Image.open(input_file).convert("RGB")
        input = np.array(input)
        input = self.transforms(input).unsqueeze(0)
        with torch.no_grad():
            log_probs = self.model(input.to(config.DEVICE))
            _, _, preds = self.decoder.decode(log_probs)
        return {"prediction": preds[0]}
        
if __name__ == "__main__":
    predictor = Predictor(str(config.MODEL_PATH/"model.pt"), str(config.DICTIONARY_PATH/"decode_dict.pkl"))
    output = predictor.predict(str(config.MAIN_PATH/"src"/"deploy"/"2b827.png"))
    print(output)