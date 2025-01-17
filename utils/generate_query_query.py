from time import time
from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import json
import logging
import pathlib, os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "ToolLens"
data_path = "./datasets/" + dataset

train_data = GenericDataLoader(data_path)
test_data = GenericDataLoader(data_path)

corpus, train , qrels = train_data.load(split="train")
corpus = train_data._load_corpus_like_query()

_, test_query , _ = train_data.load(split="test")
train_length = len(train )

print(f"train_length:{train_length}")
train_queries_1 = {}
i = 0
query_to_remove = []
for query_id, value in train.items():
    train_queries_1[query_id] = value
    query_to_remove.append(query_id)
    i = i + 1
    if i == train_length // 2:
        break
for i in query_to_remove:
    del train[i]

print(f"train_1_length:{len(train_queries_1)}")
print(f"train_2_length:{len(train)}")

model = DRES(models.SentenceBERT("PLMs/msmarco-bert-co-condensor-v1-ToolLens"), batch_size=256, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim",k_values=[1,3,5,10])

start_time = time()
results1 = retriever.retrieve(corpus, train_queries_1)
results2 = retriever.retrieve(corpus, train)
results3 = retriever.retrieve(corpus, test_query)
sorted_results={}
for key,values in results1.items():
    sorted_values = dict(sorted(values.items(), key=lambda item:item[1],reverse=True))
    sorted_results[key] = sorted_values
for key,values in results2.items():
    sorted_values = dict(sorted(values.items(), key=lambda item:item[1],reverse=True))
    sorted_results[key] = sorted_values
for key,values in results3.items():
    sorted_values = dict(sorted(values.items(), key=lambda item:item[1],reverse=True))
    sorted_results[key] = sorted_values
with open(f"./datasets/{dataset}/query_query_condensor.json", 'w') as f:
    f.write(json.dumps(sorted_results))

end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))