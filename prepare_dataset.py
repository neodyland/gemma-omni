from datasets import load_dataset
from io import BytesIO
from model.gemma_omni import GemmaOmni
import librosa
import requests
from datasets import load_dataset
import torch
import urllib
from tqdm import tqdm


def collate(model: GemmaOmni):
    b = load_dataset("saldra/sakura_japanese_dataset", split="train")
    for i, e in tqdm(enumerate(b["instruction"])):
        chat = [
            {"role": "user", "content": e},
            {"role": "assistant", "content": model.audio_token},
        ]
        wavs = [create_wav(b["output"][i])]
        with torch.inference_mode():
            yield model.encode(chat, wavs, False).squeeze(0)


def create_wav(text: str):
    res = requests.get(
        f"http://localhost:5004/?text={urllib.parse.quote(text)}&voice=98",
    )
    return librosa.load(BytesIO(res.content), sr=24000)[0]


if __name__ == "__main__":
    model = GemmaOmni()
    for i, e in enumerate(collate(model)):
        with open(f"./data/ds/{i}.txt", "w", encoding="utf8") as w:
            w.write(model.tokenizer.decode(e))
