from openai import OpenAI
from model.gemma_omni import GemmaOmni
import torch
import soundfile as sf

ai = OpenAI(base_url="http://localhost:8080", api_key="hello")
model = GemmaOmni()

res = ai.completions.create(
    model="default",
    prompt=model.tokenizer.apply_chat_template(
        [{"role": "user", "content": "おはようございます。"}],
        tokenize=False,
        add_generation_prompt=True,
    )[len(model.tokenizer.bos_token) :],
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
