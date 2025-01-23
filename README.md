# MassTool
The implementation for MassTool: A Multi-Task Search-Based Tool Retrieval Framework for Large Language Models

ToolDet is a unified dataset, which we have stored in the respective paths of each individual dataset.

## Quick Start
1. Download PLMs from Huggingface and make a folder with the name PLMs
- **ANCE**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-roberta-base-ance-firstp).
- **TAS-B**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-distilbert-base-tas-b).
- **co-Condensor**: The PLM and description is avaliable [here](https://huggingface.co/sentence-transformers/msmarco-bert-co-condensor).
- **Contriever**: The PLM and description is avaliable [here](https://huggingface.co/nthakur/contriever-base-msmarco).
2. Run Pretrained Language Models:
	> python train_sbert.py
3. Run MassTool:
	> python train.py -g 0 -m MassTool -d ToolLens -att simple -origin True -nei 10


## Environment

Our environment is conducted on Python 3.9.19
Our experimental environment is shown below:

```
numpy version: 1.25.0
transformer version: 4.39.3
torch version: 2.3.1
pandas version: 1.2.4
sentence-transformers version: 3.0.1
```

