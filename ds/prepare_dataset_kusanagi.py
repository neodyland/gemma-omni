from datasets import load_dataset
from io import BytesIO
from model.llm_omni import LLMOmni
import librosa
import requests
from datasets import load_dataset
import torch
from tqdm import tqdm
import re
from pathlib import Path
import random

RE = re.compile(r"[a-zA-Z0-9>:[\]!★▼◆♪]")


def is_ok(t):
    return not RE.match(t) and len(t) < 150 and len(t) > 20


def collate(model: LLMOmni):
    b = load_dataset("neody/kusanagi", split="train", streaming=True)
    b = b.filter(lambda x: is_ok(x["text"]))
    for e in tqdm(b["text"]):
        try:
            is_wav = random.choice([True, False])
            is_user = random.choice([True, False])
            chat = [
                {
                    "role": (
                        f"user{'_speech' if is_wav else ''}"
                        if is_user
                        else f"assistant{'_speech' if is_wav else ''}"
                    ),
                    "content": model.audio_token if is_wav else e,
                },
            ]
            wavs = [create_wav(e)] if is_wav else []
            with torch.inference_mode():
                yield model.encode(chat, wavs, False).squeeze(0)
        except Exception as err:
            print(f"Error: {err}")


def create_wav(text: str):
    res = requests.post(
        "http://localhost:5004/synthesize", json={"text": text, "ident": "iroha"}
    )
    return librosa.load(BytesIO(res.content), sr=24000)[0]


if __name__ == "__main__":
    model = LLMOmni()
    Path("./data/ds_kusanagi").mkdir(exist_ok=True, parents=True)
    for i, e in enumerate(collate(model)):
        with open(f"./data/ds_kusanagi/{i}.txt", "w", encoding="utf8") as w:
            w.write(model.tokenizer.decode(e))
