from unsloth import FastLanguageModel

if __name__ == "__main__":
    model, tokenizer = FastLanguageModel.from_pretrained(
        "./data/outputs/checkpoint-3500",
        attn_implementation="sdpa",
        device_map="cuda",
    )
    model.save_pretrained_gguf("data/gguf", tokenizer, quantization_method="q4_k_m")
