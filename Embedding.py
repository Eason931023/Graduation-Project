import os
import json
from typing import List, Dict, Any

from langchain_community.document_loaders import UnstructuredFileLoader, CSVLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- 設置與配置 ---
# 確保您已經設置了 OpenAI API Key
# os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 假設的檔案路徑
PDF_FILE_PATH = "sample_report.pdf"
CSV_FILE_PATH = "sample_data.csv"

# 向量化模型與 RAG 參數
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
FAISS_DB_PATH = "faiss_index_rag"
JSON_CACHE_PATH = "rag_cache.json"

def save_to_json_cache(data: List[Dict[str, Any]], path: str):
    """將處理後的 Document 內容儲存為 JSON 緩存。"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"✅ 資料已儲存至 JSON 緩存: {path}")

def load_from_json_cache(path: str) -> List[Document]:
    """從 JSON 緩存載入資料並轉換為 Document 物件。"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        documents.append(
            Document(
                page_content=item['page_content'],
                metadata=item['metadata']
            )
        )
    print(f"✅ 已從 JSON 緩存載入 {len(documents)} 個 Document 物件。")
    return documents