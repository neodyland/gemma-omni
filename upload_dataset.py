import os
from pathlib import Path
from datasets import Dataset

p = Path("./data/ds")
a = []
for f in os.listdir(p):
    a.append(open(p / f).read().strip())

Dataset.from_dict({"text": a}).push_to_hub("googlefan/sakura-audio")
