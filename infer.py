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
                "content": "5年生は自主的に本の寄付活動に参加しました。 5-1組は500冊、5-2組は5-1組が寄付した本の80％、5-3組は5-2組が寄付した本の120％を寄付しました。 5-1組と5-3組ではどちらが多く本を寄付したでしょうか？ (2通りで比較してください)",
            }
        ],
        [],
    )
    res = model.llm.generate(
        input_ids,
        max_new_tokens=448,
        return_dict_in_generate=False,
        streamer=TextStreamer(tokenizer=model.tokenizer, skip_prompt=True),
        repetition_penalty=2.0,
    )
    res = [x - model.audio_start_token_id for x in res.tolist()[0]]
    res_a = list(filter(lambda x: x >= 0, res))
    res_a_t = []
    for i in range(len(res_a) // 7 * 7):
        res_a_t.append(res_a[i])
    res_text = model.tokenizer.decode(
        [x + model.audio_start_token_id for x in list(filter(lambda x: x < 0, res))],
        skip_special_tokens=True,
    )
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
