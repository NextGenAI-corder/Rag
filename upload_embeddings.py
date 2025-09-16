import os
import fitz  # PyMuPDF
import openai
import tiktoken
from pinecone import Pinecone

# ▼ APIキーは環境変数から取得（直接書いても可）
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ▼ Pinecone v3クライアント初期化
pc = Pinecone(api_key=pinecone_api_key)

# ▼ 接続するインデックス名（事前に作成済みであること）
index = pc.Index("urata-soft")

# ▼ テキストをトークン数で分割（500トークン前後）
def split_text(text, max_tokens=500):
    encoding = tiktoken.get_encoding("cl100k_base")
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        test = current + "\n" + line
        if len(encoding.encode(test)) <= max_tokens:
            current = test
        else:
            chunks.append(current.strip())
            current = line
    if current:
        chunks.append(current.strip())
    return chunks

# ▼ PDF読み込み→Embedding化→Pineconeアップロード
def process_pdf(pdf_path, namespace="nextgen-specs"):
    print(f"\n📄 処理開始: {pdf_path}")
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()

    chunks = split_text(full_text)

    for i, chunk in enumerate(chunks):
        embedding = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        ).data[0].embedding

        index.upsert(
            vectors=[{
                "id": f"{os.path.basename(pdf_path)}-chunk-{i}",
                "values": embedding,
                "metadata": {"text": chunk}
            }],
            namespace=namespace
        )
        print(f"✅ {pdf_path} → chunk {i} アップロード完了")

# ▼ 対象のPDFファイルをここに列挙（ファイル名を正確に）
pdf_files = [
    "./PDF/ourCo.pdf"
]

# ▼ 実行
for file in pdf_files:
    process_pdf(file)
