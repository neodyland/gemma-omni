from model.gemma_omni import GemmaOmni
from transformers import TextStreamer
import torch
import soundfile as sf

if __name__ == "__main__":
    model = GemmaOmni("./data/outputs/checkpoint-126")

    input_ids = model.encode(
        [
            {
                "role": "user",
                "content": "こんにちは",
            }
        ],
        [],
    )
    res = model.llm.generate(
        input_ids,
        max_new_tokens=448,
        return_dict_in_generate=False,
        streamer=TextStreamer(tokenizer=model.tokenizer, skip_prompt=True),
        repetition_penalty=1.2,
    )
    res = [x - model.audio_start_token_id for x in res.tolist()[0][input_ids.size(1) :]]
    res_a = list(filter(lambda x: x >= 0, res))
    res_a_t = []
    for i in range(len(res_a) // 7 * 7):
        res_a_t.append(res_a[i])
    res_text = model.tokenizer.decode(
        [x + model.audio_start_token_id for x in list(filter(lambda x: x < 0, res))],
        skip_special_tokens=True,
    )
    print(res_text)
    with torch.inference_mode():
        sf.write(
            "data/out.wav",
            data=model.snac.decode(torch.tensor(res_a_t).unsqueeze(0).to(model.device))
            .squeeze(0)
            .cpu()
            .float()
            .numpy()
            .T,
            samplerate=model.snac.sr,
        )
