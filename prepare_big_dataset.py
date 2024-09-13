from datasets import load_dataset, VerificationMode
from io import BytesIO
from model.gemma_omni import GemmaOmni
import librosa
import requests
from datasets import load_dataset
import torch
import urllib
from tqdm import tqdm
import re

RE = re.compile(r"[a-zA-Z0-9>:]")


def is_ok(t, e=True):
    return not RE.match(t) and len(t) < 150 and (len(t) > 20 if e else True)


def collate(model: GemmaOmni):
    b = load_dataset(
        "Aruno/guanaco_jp", split="train", verification_mode=VerificationMode.NO_CHECKS
    )
    b = b.filter(
        lambda x: is_ok(x["input"], False)
        and is_ok(x["instruction"])
        and is_ok(x["output"])
    )
    for i, e in enumerate(tqdm(b["instruction"])):
        chat = [
            {
                "role": "user",
                "content": e + ("\n" + b["input"][i]) if len(b["input"][i]) > 0 else "",
            },
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
        with open(f"./data/ds_big/{i}.txt", "w", encoding="utf8") as w:
            w.write(model.tokenizer.decode(e))
