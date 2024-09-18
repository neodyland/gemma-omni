import torch
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast, BitsAndBytesConfig
from model.snac_gasi import SnacGasi

llm: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    ),
)
tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct"
)
snac = SnacGasi()
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant_speech\n' }}{% endif %}"
tokenizer.add_tokens([f"<audio_token_{i + 1}>" for i in range(snac.size)])
llm.resize_token_embeddings(len(tokenizer))
llm.save_pretrained("./data/llm")
tokenizer.save_pretrained("./data/llm")
