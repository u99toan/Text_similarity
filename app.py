import numpy as np
from sklearn.cluster import KMeans
import json

import uvicorn
from pydantic import BaseModel, conlist
from fastapi import FastAPI
from typing import List

from utils import load_data, standardize_data, vector_emb, embed_text, emb_text
from sklearn.metrics.pairwise import cosine_similarity

# save_data, list_id, list_text = load_data("data.csv")

app = FastAPI(
    title="Text New Similarity",
    description="A simple API that use NLP model check Similarity",
    version="0.1",
)

class text_sample(BaseModel):
    id : str
    text: str

class batch(BaseModel):
    #batchs: List[conlist(item_type=float, min_items=1, max_items=20)]
    list_item : List[text_sample]

@app.post("/cluster-paper")
async def predict_batch(item: batch, num):
    list_data = item.list_item
    list_text = []
    list_id = []
    for data in list_data:
        id = data.id
        text = data.text
        list_id.append(id)
        list_text.append(text)
    emb_vec = [vector_emb(list_text[i]) for i in range(len(list_text))]
    cluster = KMeans(init="k-means++", n_clusters=3, n_init=5)
    

    cluster.fit(emb_vec)
    y_hat = cluster.predict(emb_vec)

    list_paper = []
    for idx in y_hat:
        if idx not in list_paper:
            list_paper.append(idx)



    # pair = zip(y_hat, list_id)

    # values, counts = np.unique(y_hat, return_counts=True)

    # counts = np.argsort(counts)
    # counts = counts[::-1]
    # find = counts[0]

    # list_id_output = []
    # for y_hat, id in pair:
    #     if y_hat == values[find]:
    #         list_id_output.append(id)
    
    k = int(num)
    if len(list_paper) >= k:
        out = [[{"id": str(i)}, {"text": str(list_text[i])}]  for i in list_paper[:k]]
    else:
        out = [[{"id": str(i)}, {"text": str(list_text[i])}]  for i in list_paper]


    output = json.loads(json.dumps(out))
    return output



# @app.get("/predict-text")
# async def predict_text(text: str, num: str):
#     vec = emb_text(text)
#     vec = np.array(vec)
#     vec = vec.reshape(1,-1)
#     sim_scores = cosine_similarity(vec, save_data)
#     score = sim_scores[0]
#     pair = zip(list_id, score)                                                          

#     sim_scores = sorted(pair, key=lambda x: x[1], reverse=True)
#     k = int(num)
#     sim_scores = sim_scores[0:k]

#     list_out = []
#     for id in sim_scores:
#         list_out.append(list_text[id[0]])
#     return str(list_out)

class text_class(BaseModel):
    text_a : str
    text_b: str

@app.post("/similar-text")
def score_2_vector(text: text_class):
    vec_1 = emb_text(text.text_a)
    vec_2 = emb_text(text.text_b)
    vec_1 = vec_1.reshape(1,-1)
    vec_2 = vec_2.reshape(1,-1)


    sim_scores = cosine_similarity(vec_1, vec_2)
    score = sim_scores[0]
    return str(score)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4501)
