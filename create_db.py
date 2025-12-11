import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# --- 配置參數 ---
# 1. 您的 PDF 檔案名稱 (根據您先前的截圖，可能是 'llm.pdf')
PDF_PATH = "llm.pdf" # 建議您再次確認此處名稱是否正確
# 2. Ollama 上使用的嵌入模型名稱 (nomic-embed-text)
EMBEDDING_MODEL = "nomic-embed-text"
# 3. 向量資料庫儲存路徑
DB_FAISS_PATH = "vectorstore/db_faiss"
# 4. 文本分割參數
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_vector_db():
    """
    載入 PDF, 分割文本, 建立 Ollama 向量嵌入, 並儲存到 FAISS 資料庫。
    """
    print(f"--- 開始處理 PDF: {PDF_PATH} ---")
    
    # 檢查 PDF 檔案是否存在
    if not os.path.exists(PDF_PATH):
        print(f"❌ 錯誤：找不到檔案 {PDF_PATH}。請確認檔案路徑是否正確。")
        return

    try:
        # 1. 載入 PDF 文件
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"✅ 成功載入 {len(documents)} 頁文件。")

        # 2. 分割文本
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        texts = text_splitter.split_documents(documents)
        print(f"✅ 文件分割成 {len(texts)} 個文本塊 (chunks)。")

        # 3. 建立 Ollama 嵌入模型實例
        # 使用 nomic-embed-text 進行向量化
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        print(f"✅ 正在使用 Ollama 嵌入模型: {EMBEDDING_MODEL}")

        # 4. 建立 FAISS 向量資料庫
        print("⏳ 正在建立 FAISS 向量資料庫，這可能需要幾分鐘...")
        vectorstore = FAISS.from_documents(texts, embeddings)
        print("✅ 向量資料庫建立完成。")
        
        # 5. 儲存資料庫到本地磁碟
        # 確保儲存路徑存在
        os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
        vectorstore.save_local(DB_FAISS_PATH)
        print(f"💾 資料庫已成功儲存至: {DB_FAISS_PATH}")
        print("--- 資料庫建立完成，現在可以運行您的 QA 程式了！ ---")


    except Exception as e:
        print(f"❌ 處理過程中發生錯誤: {e}")

if __name__ == "__main__":
    create_vector_db()