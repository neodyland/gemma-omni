from datasets import load_dataset, VerificationMode
from io import BytesIO
from model.llm_omni import LLMOmni
import librosa
import requests
from datasets import load_dataset
import torch
from tqdm import tqdm
import re
from pathlib import Path

RE = re.compile(r"[a-zA-Z0-9>:[\]!]")


def is_ok(t, e=True):
    return not RE.match(t) and len(t) < 150 and (len(t) > 20 if e else True)


def collate(model: LLMOmni):
    b = load_dataset(
        "Aruno/guanaco_jp", split="train", verification_mode=VerificationMode.NO_CHECKS
    )
    b = b.filter(
        lambda x: is_ok(x["input"], False)
        and is_ok(x["instruction"])
        and is_ok(x["output"])
    )
    for i, e in enumerate(tqdm(b["instruction"])):
        try:
            chat = [
                {
                    "role": "user",
                    "content": e
                    + (("\n" + b["input"][i]) if len(b["input"][i]) > 0 else ""),
                },
                {"role": "speech", "content": model.audio_token},
            ]
            wavs = [create_wav(b["output"][i])]
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
    Path("./data/ds").mkdir(exist_ok=True, parents=True)
    for i, e in enumerate(collate(model)):
        with open(f"./data/ds/{i}.txt", "w", encoding="utf8") as w:
            w.write(model.tokenizer.decode(e))
