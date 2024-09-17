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
tokenizer.chat_template = tokenizer.chat_template.replace("assistant", "speech")
tokenizer.add_tokens([f"<audio_token_{i + 1}>" for i in range(snac.size)])
llm.resize_token_embeddings(len(tokenizer))
llm.save_pretrained("./data/llm")
tokenizer.save_pretrained("./data/llm")
