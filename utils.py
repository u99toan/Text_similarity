import re
import pandas as pd
from tqdm import tqdm
import numpy as np
from pyvi.ViTokenizer import tokenize
from sentence_transformers import SentenceTransformer


def load_data(path_data):
    list_text = []
    list_id_text = []
    df = pd.read_csv(path_data).fillna(' ')
    list_id = []
    save_data = []
    cnt = 0

    for index, row in tqdm(df.iterrows()):
        item = {
            'id': row['id'],
            'text': row['text']
        }
        list_text.append(item["text"])
        emb = vector_emb(item["text"])
        list_id.append(cnt)
        save_data.append(emb)
        cnt+=1


    save_data = np.array(save_data)
    list_id = np.array(list_id)
    return save_data, list_id, list_text


def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\,\?]+$-()!*=._", "", row)
    row = row.replace(",", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("*", " ")\
        .replace("=", " ").replace("(", " ")\
        .replace(")", " ").replace("_", " ").replace(".", " ")
    row = row.strip().lower()
    return row

def embed_text(text):
    index_name = "demo_simcse"
    path_index = "config/index.json"
    model_embedding = SentenceTransformer('VoVanPhuc/sup-SimCSE-VietNamese-phobert-base')
    emb = model_embedding.encode(text)
    return emb

def emb_text(text):
    stand = standardize_data(text)
    titles = tokenize(stand)
    title_vectors = embed_text(titles)
    return title_vectors
def vector_emb(text):
    stand = standardize_data(text)
    titles = tokenize(stand)
    title_vectors = embed_text(titles)
    return title_vectors