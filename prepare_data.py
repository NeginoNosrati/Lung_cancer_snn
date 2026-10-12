import json, hashlib, shutil, random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split


SEED = 42
SRC_ROOT = Path.home() / "datasets" / "IQ-OTHNCCD"
DST = Path("../data")
EXTS = {".jpg", ".jpeg", ".png"}
random.seed(SEED)
np.random.seed(SEED)


def class_of(folder_name: str):
    n = folder_name.lower()
    if "ben" in n:   return "benign"     
    if "malig" in n: return "malignant"
    if "normal" in n: return "normal"
    return None
    
print(class_of("bengin_cases"))
print(class_of("Malignant cases"))
print(class_of("Normal cases"))    
print(class_of("Test cases"))               
    


records = []
for d in SRC_ROOT.rglob("*"):
    if d.is_dir():
        cls = class_of(d.name)
        if cls is None:
            continue
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in EXTS:
                records.append({"path": str(f), "class": cls})

df = pd.DataFrame(records)
assert len(df) > 0, "no pic found"
print( ":before removing duplicates\n", df["class"].value_counts(), "\n")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()

df["md5"] = df["path"].apply(md5)
dups = df[df.duplicated("md5", keep="first")]
print(f"removed dups numbers: {len(dups)}")
df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)


sizes, modes = Counter(), Counter()
for p in df["path"]:
    with Image.open(p) as im:
        sizes[im.size] += 1
        modes[im.mode] += 1
print("\n scales:", sizes.most_common(5))
print("colors :", dict(modes))



train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["class"], random_state=SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["class"], random_state=SEED)

train_df = train_df.assign(split="train")
val_df = val_df.assign(split="val")
test_df = test_df.assign(split="test")
all_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
all_df.sample(10, random_state=42)




for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
    A = set(all_df[all_df.split == a]["md5"]); B = set(all_df[all_df.split == b]["md5"])
    assert not (A & B), f"OVERLAP NETWEEN{a} و {b}!"
print("\n NO DATA LEAKAGE FOUND")



for i, (_, r) in enumerate(all_df.iterrows()):
    if i % 100 == 0:
        print(f"  {i}/{len(all_df)}")
    
    out = DST / r["split"] / r["class"]
    out.mkdir(parents=True, exist_ok=True)
    
    src = Path(r["path"])
    dst_file = out / f"{r['md5'][:8]}_{src.name}"
    
    try:
        shutil.copy(src, dst_file)
    except Exception as e:
        print(f" eror: {src} → {e}")

all_df.to_csv("../split_manifest.csv", index=False)


all_df.to_csv("split_manifest.csv", index=False)


table = pd.crosstab(all_df["class"], all_df["split"], margins=True)[["train", "val", "test", "All"]]
print(table)
print((pd.crosstab(all_df["split"], all_df["class"], normalize="index") * 100).round(1))



classes = sorted(train_df["class"].unique())
counts = train_df["class"].value_counts()
weights = {c: float(len(train_df) / (len(classes) * counts[c])) for c in classes}
json.dump({"classes": classes, "weights": weights}, open("class_weights.json", "w"), indent=2)


for c, w in weights.items():
    print(f"  {c}: {w:.3f}")




fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i, c in enumerate(classes):
    samples = train_df[train_df["class"] == c].sample(min(3, len(train_df[train_df["class"] == c])), random_state=SEED)["path"]
    for j, p in enumerate(samples):
        try:
            axes[i, j].imshow(Image.open(p).convert("L"), cmap="gray")
            axes[i, j].set_title(c, fontsize=10)
        except:
            axes[i, j].text(0.5, 0.5, "error", ha="center", va="center")
        axes[i, j].axis("off")

plt.tight_layout()
plt.savefig("samples.png", dpi=150)
plt.show()
