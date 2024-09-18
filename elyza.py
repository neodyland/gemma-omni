from openai import OpenAI
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

ai = OpenAI(base_url="http://localhost:8080/v1", api_key="hello")
ai2 = OpenAI(base_url="http://localhost:8081/v1", api_key="hello")
tokenizer = AutoTokenizer.from_pretrained("./data/llm")


def generate_response(q: str):
    res = ai.completions.create(
        model="default",
        prompt=tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False,
            add_generation_prompt=True,
        )[len(tokenizer.bos_token) if tokenizer.bos_token else 0 :].replace(
            "assistant_speech", "assistant"
        ),
        extra_body={
            "repeat_penalty": 1.2,
        },
        max_tokens=1000,
    )
    return res.content


def generate_score(p: str):
    res = ai2.chat.completions.create(
        model="default",
        messages=[{"role": "user", "content": p}],
        extra_body={"grammar": "root ::= [1-5]"},
    )
    return float(res.choices[0].message.content)


def eval_one(q: str, a: str, aspect: str):
    pred = generate_response(q)
    res = generate_score(
        f"""あなたは採点者です。

問題, 正解例, 採点基準, 回答 が与えられます。

採点基準と正解例を参考にして、回答を1,2,3,4,5の5段階で採点し、数字のみを出力してください。

# 問題
{q}

# 正解例
{a}

# 採点基準
基本的な採点基準
- 1点: 誤っている、 指示に従えていない
- 2点: 誤っているが、方向性は合っている
- 3点: 部分的に誤っている、 部分的に合っている
- 4点: 合っている
- 5点: 役に立つ

基本的な減点項目
- 不自然な日本語: -1点
- 部分的に事実と異なる内容を述べている: -1点
- 「倫理的に答えられません」のように過度に安全性を気にしてしまっている: 2点にする

問題固有の採点基準
{aspect}

# 回答
{pred}"""
    )
    return res, pred


if __name__ == "__main__":
    ds = load_dataset("elyza/ELYZA-tasks-100", split="test")
    score = 0.0
    res = []
    for entry in tqdm(ds):
        q = entry["input"]
        a = entry["output"]
        aspect = entry["eval_aspect"]
        s, pred = eval_one(
            q,
            a,
            aspect,
        )
        score += s
        print(f"Score: {s}\nQuestion: {q}\nAnswer: {a}\nPred: {pred}")
        res.append(
            {"score": int(s), "question": q, "pure_answer": a, "pred_answer": pred}
        )
    print(f"Score: {score / len(ds)}")
