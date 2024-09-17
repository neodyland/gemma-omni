import torch
from transformers import Qwen2ForCausalLM, Qwen2TokenizerFast
from model.snac_gasi import SnacGasi

llm: Qwen2ForCausalLM = Qwen2ForCausalLM.from_pretrained(
    "unsloth/Qwen2-1.5B-bnb-4bit", torch_dtype=torch.bfloat16
)
tokenizer: Qwen2TokenizerFast = Qwen2TokenizerFast.from_pretrained(
    "unsloth/Qwen2-1.5B-bnb-4bit"
)
snac = SnacGasi()
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>audio\n' }}{% endif %}"
tokenizer.add_tokens([f"<audio_token_{i + 1}>" for i in range(snac.size)])
llm.resize_token_embeddings(len(tokenizer))
llm.save_pretrained("./data/llm")
tokenizer.save_pretrained("./data/llm")
