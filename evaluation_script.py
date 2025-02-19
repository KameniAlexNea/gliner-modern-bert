# %%
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["WANDB_PROJECT"] = "gliner_finetuning"
# os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "none"

# %%
from gliner import GLiNER

# model = GLiNER.from_pretrained("data/models/checkpoint-100000")
model = GLiNER.from_pretrained("data/models/checkpoint-90288")
model.cuda()

# %%
prob_min = 0.1

# %%
from glob import glob
import json

test_files = glob("data/IE_INSTRUCTIONS/NER/*/test.json")
print(len(test_files))

# %%
file = test_files[0]
print(file)

with open(file) as f:
	raws: list[dict] = json.load(f)
	
raws[0]

# %%
labels=list(set(ent["type"] for ent in raws[0]["entities"]))

print(labels)

model.predict_entities(raws[0]["sentence"], labels=labels)

# %%
os.path.dirname(file)

# %%
all_labels = json.load(open(os.path.join(
	os.path.dirname(file), "labels.json"
)))

all_labels

# %%
raws[:10]

# %%
texts = [raw["sentence"] for raw in raws[:10]]

texts

# %%
preds = model.batch_predict_entities(texts, labels=all_labels)

for pred in preds:
	print(pred)
	print()

# %%
model.predict_entities(texts[-2], labels=all_labels)

# %%
raws[8]

# %%
all_labels

# %%
texts[-2]

# %%
model.predict_entities(texts[-2], labels=["country"])

# %%
def evaluate_ner(true_entities, predicted_entities):
	tp, fp, fn = 0, 0, 0

	true_set = {(ent['pos'][0], ent['pos'][1], ent['type']) for ent in true_entities}
	pred_set = {(ent['start'], ent['end'], ent['label']) for ent in predicted_entities}

	tp = len(true_set & pred_set)  # Intersection: Correctly predicted
	fp = len(pred_set - true_set)  # Predicted but not in ground truth
	fn = len(true_set - pred_set)  # Ground truth but not predicted

	precision = tp / (tp + fp) if (tp + fp) > 0 else 0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0
	f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

	return {"Precision": precision, "Recall": recall, "F1-score": f1_score}

# %%
raws[0]["entities"]

# %%
preds[0]

# %%
evaluate_ner(raws[0]["entities"], preds[0])

# %%
from tqdm import tqdm

# %%
import pandas as pd

batch_size = 16
thr = 0.1

save_folder = "data/NEREvalsLast"
os.makedirs(save_folder, exist_ok=True)

for file in test_files:
	all_predictions = []
	all_expected = []
	with open(file, encoding="utf-8") as f:
		raws: list[dict] = json.load(f)
	file_folder = os.path.dirname(file)
	name = os.path.basename(file_folder)
	all_labels = json.load(open(os.path.join(file_folder, "labels.json")))

	scores = []
	for i in tqdm(range(0, len(raws), batch_size)):
		texts = [raw["sentence"] for raw in raws[i : i + batch_size]]
		expected = [raw["entities"] for raw in raws[i : i + batch_size]]
		preds = model.batch_predict_entities(texts, labels=all_labels, threshold=thr)
		all_predictions.append(preds)
		all_expected.append(expected)
	scores = [
		evaluate_ner(j1, j2)
		for i1, i2 in zip(all_expected, all_predictions)
		for j1, j2 in zip(i1, i2)
	]
	print(name)
	print(pd.DataFrame(scores).mean())
	all_predictions = [j for i in all_predictions for j in i]
	with open(f"{save_folder}/{name}.json", "w") as f:
		json.dump(all_predictions, f)
	print()

# %%



