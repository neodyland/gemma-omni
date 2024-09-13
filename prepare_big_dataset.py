from datasets import load_dataset
import os
import requests
from transformers import GemmaTokenizerFast
import re

RE_ENG = re.compile(r"[a-zA-Z]{2,}|訳")

tok: GemmaTokenizerFast = GemmaTokenizerFast.from_pretrained(
    "unsloth/gemma-2-9b-it-bnb-4bit"
)

cpus = os.cpu_count() * 3 // 4


def is_ok(s: str):
    return all(
        [
            x not in s
            for x in [
                "^",
                "$",
                "frac",
                "\\",
                "-",
                "+",
                "/",
                "*",
                "=",
                "[",
                "]",
                "(",
                ")",
            ]
        ]
    )


def collate():
    b = load_dataset("chargoddard/WebInstructSub-prometheus", split="train")
    b = b.rename_columns({"instruction": "user", "generation": "assistant"})
    b = b.select_columns(["user", "assistant"])
    b = b.filter(
        lambda b: is_ok(b["user"]) and is_ok(b["assistant"]),
        batched=False,
        num_proc=cpus,
    )
    b = b.map(
        lambda x: translate(x),
        batched=False,
        num_proc=cpus * 40,
    )
    b = b.filter(
        lambda b: b["user"] != "" and b["assistant"] != "",
        batched=False,
        num_proc=cpus,
    )
    return b


def translate(pair):
    x = translate__(pair["user"])
    if x == "":
        print("Failed")
        return {
            "user": "",
            "assistant": "",
        }
    y = translate__(pair["assistant"])
    if y == "":
        print("Failed")
    return {
        "user": x,
        "assistant": y,
    }


def translate__(q: str):
    res = ""
    c = 0
    while res == "" and c < 3:
        res = translate_(q)
        c = c + 1
    return res


def translate_(q: str):
    try:
        res = (
            requests.post(
                "http://localhost:8000/v1/completions",
                json={
                    "prompt": tok.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"{q}\n日本語の話し言葉に訳し、結果だけを出力してください。",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )[len(tok.bos_token) :],
                    "model": "default",
                    "max_tokens": 512,
                },
            )
            .json()["choices"][0]["text"]
            .strip()
        )
        if RE_ENG.match(res):
            return ""
        return res or ""
    except:
        return ""


if __name__ == "__main__":
    ds = collate()
    print(len(ds))
    ds.push_to_hub("googlefan/WebInstructSub-ja")
