from model.gemma_omni import GemmaOmni
from transformers import TextStreamer
import torch
from unsloth import FastLanguageModel

if __name__ == "__main__":
    model = GemmaOmni()
    model.tokenizer.chat_template = model.tokenizer.chat_template.replace(
        "audio", "model"
    )
    llm, _ = FastLanguageModel.from_pretrained(
        "./data/outputs/checkpoint-2500",
        attn_implementation="sdpa",
        device_map="cuda",
    )
    FastLanguageModel.for_inference(llm)
    input_ids = model.encode(
        [
            {
                "role": "user",
                "content": "おはようございます。",
            }
        ],
        [],
    ).to(llm.device)
    with torch.inference_mode():
        res = llm.generate(
            input_ids,
            max_new_tokens=2563,
            return_dict_in_generate=False,
            streamer=TextStreamer(tokenizer=model.tokenizer, skip_prompt=True),
            repetition_penalty=1.2,
        )
    res_text = model.tokenizer.decode(
        res.squeeze()[input_ids.size(1) :],
        skip_special_tokens=True,
    )
    print(res_text)