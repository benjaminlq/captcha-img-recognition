import config
from dev import utils
from fastapi import APIRouter, File, UploadFile
from deploy.inference import Predictor

model_path = str(config.MODEL_PATH/"model.pt")
decoder_path = str(config.DICTIONARY_PATH/"decode_dict.pkl")

predictor = Predictor(model_path = model_path, decoder_path = decoder_path)

router = APIRouter(prefix = "/captcha")

@router.get("/")
def service_description():
    return {"Service":"Translate Captcha Images to Text"}

@router.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        output = predictor.predict(file.file)
        return output

    except Exception:
        return {"Error":"Error Uploading File"}

    finally:
        file.file.close()