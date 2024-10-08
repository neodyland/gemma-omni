from openai import OpenAI
from model.llm_omni import LLMOmni
import torch
import soundfile as sf
import os

speech = False
ai = OpenAI(base_url="http://localhost:8080", api_key="hello")
model = LLMOmni()

res = ai.completions.create(
    model="default",
    prompt=model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "LLMについて教えて"}],
        tokenize=False,
        add_generation_prompt=True,
    )[len(model.tokenizer.bos_token) if model.tokenizer.bos_token else 0 :].replace(
        "assistant_speech", "assistant_speech" if speech else "assistant"
    ),
    stream=True,
    extra_body={
        "repeat_penalty": 1.2,
    },
)
last = ""
for chunk in res:
    c = chunk.content
    if c:
        last += c
    print(c if c else "\n", end="", flush=True)
if not speech:
    os._exit(0)
with torch.inference_mode():
    ts = torch.tensor(
        [
            x - model.audio_start_token_id
            for x in model.tokenizer(last).input_ids
            if x - model.audio_start_token_id >= 0
        ]
    ).to(model.device)
    ts = ts[: ts.size(0) // 7 * 7]
    sf.write(
        "data/out.wav",
        data=model.snac.decode(ts.unsqueeze(0)).squeeze(0).cpu().float().numpy().T,
        samplerate=model.snac.sr,
    )
