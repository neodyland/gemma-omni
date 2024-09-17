from torch import nn
from transformers import Qwen2TokenizerFast
import torch
from .snac_gasi import SnacGasi
from typing import List, Dict
import numpy as np


class LLMOmni(nn.Module):
    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.snac = SnacGasi()
        self.audio_token = "<|speech|>"
        self.tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained(
            "./data/llm"
        )
        self.audio_start_token_id = self.tokenizer.convert_tokens_to_ids(
            "<audio_token_1>"
        )
        self.device = device
        self.dtype = torch.bfloat16
        self.to(self.device, dtype=self.dtype)

    def encode(
        self,
        conv: List[Dict[str, str]],
        audios: List[np.ndarray],
        add_generation_prompt=True,
    ):
        prompt = self.tokenizer.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=add_generation_prompt
        )
        parts = []
        audios = [
            self.snac.encode(
                torch.from_numpy(audio)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device=self.device, dtype=self.dtype, non_blocking=True)
            )
            for audio in audios
        ]
        for i, spl in enumerate(prompt.split(self.audio_token)):
            if i != 0:
                parts.append(audios[i - 1] + self.audio_start_token_id)
            parts.append(
                self.tokenizer.encode(
                    spl, add_special_tokens=False, return_tensors="pt"
                ).to(device=self.device)
            )
        input_ids = torch.cat(parts, dim=1)
        return input_ids


if __name__ == "__main__":
    model = LLMOmni()
    import librosa
    from unsloth import FastLanguageModel

    llm = FastLanguageModel.from_pretrained("./data/llm")

    audio, _sr = librosa.load("./data/wavs/vicuna_1.wav", sr=model.snac.sr)
    input_ids = model.encode(
        [
            {
                "role": "user",
                "content": "<|speech|>おはよう",
            }
        ],
        [audio],
    )
    with torch.inference_mode():
        res = llm.generate(
            input_ids, max_new_tokens=1024, return_dict_in_generate=False
        )
    res_text = model.tokenizer.decode(
        res.squeeze(0)[input_ids.size(1) :], skip_special_tokens=True
    )
    print(res_text)
