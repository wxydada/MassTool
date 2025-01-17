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
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

dataset = "ToolLens"
data_path = "./datasets/" + dataset

corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

max_tool = len(corpus)

modify_corpus, modify_queries, modify_qrels = GenericDataLoader(data_folder=data_path,query_file="new_queries.jsonl").load(split="test")

model = DRES(models.SentenceBERT("PLMs/msmarco-roberta-base-ance-firstp-v1-ToolLens-new"), batch_size=256, corpus_chunk_size=512*9999)
retriever = EvaluateRetrieval(model, score_function="cos_sim",k_values=[1,3,5,10])

start_time = time()

results = retriever.retrieve(corpus, queries)
# results_1 = retriever.retrieve(corpus, queries)
# results_2 = retriever.retrieve(corpus, modify_queries)
# results = {}
# for id,tools in results_1.items():
#     tool_list = tools
#     tool_list = dict(sorted(tool_list.items(), key=lambda item: int(item[0])))
#     results_1[id] = tool_list

# for id,tools in results_2.items():
#     tool_list = tools
#     tool_list = dict(sorted(tool_list.items(), key=lambda item: int(item[0])))
#     results_2[id] = tool_list

# for id,tools in results_1.items():
#     tools_2 = results_2[id]
#     for key,value in tools.items():
#         tools[key] = (value + tools_2[key]) / 2
#     tools = dict(sorted(tools.items(), key=lambda item: item[1]))
#     results[id] = tools

end_time = time()
print("Time taken to retrieve: {:.2f} seconds".format(end_time - start_time))
#### Evaluate your retrieval using NDCG@k, MAP@K ...

logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10])

mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="r_cap")
hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")

