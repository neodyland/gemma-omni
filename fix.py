import torch
from transformers import Gemma2ForCausalLM, GemmaTokenizerFast
from model.snac_gasi import SnacGasi

llm: Gemma2ForCausalLM = Gemma2ForCausalLM.from_pretrained(
    "unsloth/gemma-2-2b-it-bnb-4bit", torch_dtype=torch.bfloat16
)
tokenizer: GemmaTokenizerFast = GemmaTokenizerFast.from_pretrained(
    "unsloth/gemma-2-2b-it-bnb-4bit"
)
snac = SnacGasi()
tokenizer.chat_template = tokenizer.chat_template.replace("model", "audio")
tokenizer.add_tokens([f"<audio_token_{i + 1}>" for i in range(snac.size)])
llm.resize_token_embeddings(len(tokenizer))
llm.save_pretrained("./data/llm")
tokenizer.save_pretrained("./data/llm")
