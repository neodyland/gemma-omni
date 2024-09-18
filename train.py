from unsloth import (
    FastLanguageModel,
    is_bfloat16_supported,
    UnslothTrainer,
    UnslothTrainingArguments,
)
from datasets import load_dataset
import torch

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True

max_seq_length = 5120
ds = load_dataset("googlefan/kusanagi-audio", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./data/llm",
    max_seq_length=max_seq_length,
    dtype=None,
    attn_implementation="sdpa",
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
        "embed_tokens",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None,
)

trainer = UnslothTrainer(
    model=model,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    packing=True,
    args=UnslothTrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.01,
        num_train_epochs=5,
        learning_rate=1e-4,
        embedding_learning_rate=3e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="data/outputs-pretrained",
        optim="adamw_8bit",
        seed=3407,
        save_total_limit=2,
        save_steps=250,
    ),
)
trainer.train()
