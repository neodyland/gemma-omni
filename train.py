import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

max_seq_length = 2048
ds = load_dataset("googlefan/sakura-audio", split="train")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./data/llm",
    max_seq_length=max_seq_length,
    dtype=None,
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
        "*",
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

trainer = SFTTrainer(
    model=model,
    train_dataset=ds,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        num_train_epochs=5,
        learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        output_dir="data/outputs",
        optim="adamw_8bit",
        seed=3407,
    ),
)
trainer.train()
