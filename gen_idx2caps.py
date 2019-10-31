import json

with open('data/dataset_coco.json', 'r') as f:
    dataset = json.load(f)

idx2caps = {}
for data in dataset['images']:
    idx = int(data['cocoid'])
    idx2caps[idx] = [s['raw'] for s in data['sentences']]

with open('data/idx2caps.json', 'w') as f:
    json.dump(idx2caps, f)
