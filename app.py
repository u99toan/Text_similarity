import pandas as pd
from sentence_transformers import SentenceTransformer
from pyvi.ViTokenizer import tokenize
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
import re
import json

import uvicorn
from pydantic import BaseModel, conlist
from fastapi import FastAPI
from typing import List