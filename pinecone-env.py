import os
from pinecone import Pinecone

# Pinecone APIキーで初期化
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# 既存インデックスへ接続
index = pc.Index("urata-soft")
