from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- 配置參數 (已更新為 gemma3:4b 和 nomic-embed-text) ---
LLM_MODEL = "gemma3:4b"               # Ollama 上用於問答的 LLM 模型 (保持不變)
EMBEDDING_MODEL = "nomic-embed-text"  # <-- 已修改：Ollama 上用於嵌入的模型
DB_FAISS_PATH = "vectorstore/db_faiss" # 向量資料庫儲存路徑
K_RETRIEVAL = 4                       # 檢索器要找回的最相關文本塊數量

def initialize_qa_chain():
    """
    載入向量資料庫，初始化 Ollama LLM 和 RAG 鏈。
    """
    # 檢查資料庫檔案是否存在
    if not os.path.exists(DB_FAISS_PATH):
        # 提醒用戶需要先建立資料庫
        print(f"錯誤：找不到向量資料庫資料夾 {DB_FAISS_PATH}。請先運行 create_db.py，並確認使用 {EMBEDDING_MODEL} 建立。")
        return None

    print(f"--- 初始化 QA Chain，使用 LLM: {LLM_MODEL} ---")
    
    # 1. 載入嵌入模型 (用於將問題轉換成向量)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    print(f"✅ 載入嵌入模型: {EMBEDDING_MODEL}。")

    # 2. 載入 FAISS 向量資料庫
    # 注意：這裡載入的資料庫必須是使用 nomic-embed-text 創建的
    vectorstore = FAISS.load_local(
        DB_FAISS_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 3. 建立檢索器 (Retriever)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K_RETRIEVAL})
    print(f"✅ 載入向量資料庫，設置檢索器 (K={K_RETRIEVAL})。")

    # 4. 初始化本地 Ollama LLM
    llm = Ollama(model=LLM_MODEL, temperature=0.1) 
    print(f"✅ 載入 Ollama LLM 模型: {LLM_MODEL}。")

    # 5. 定義 RAG 提示詞模板 (Prompt Template)
    custom_prompt_template = """
    你是一位專業的文件分析助理。請僅根據提供的上下文 (Context) 來回答問題 (Question)。
    如果上下文沒有包含足以回答問題的資訊，請禮貌地回答「根據提供的文件內容，我無法找到相關資訊來回答這個問題」。

    Context: {context}

    Question: {question}

    有用的回答:
    """
    
    # 6. 建立 QA 鏈
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])}
    )
    
    print("--- 初始化完成，您可以開始提問了！ ---")
    return qa_chain

def main():
    """
    主循環，讓使用者可以不斷輸入問題。
    """
    qa_chain = initialize_qa_chain()
    
    if qa_chain is None:
        return

    while True:
        query = input("\n請輸入您的問題 (或輸入 'quit' 退出): ")
        if query.lower() == 'quit':
            print("再見！QA Chain 已退出。")
            break
        
        if not query.strip():
            continue

        print("⏳ QA Chain 正在思考中...")
        
        # 執行 RAG 鏈
        response = qa_chain.invoke({"query": query})
        
        # 輸出結果
        print("\n" + "="*50)
        print(f"❓ 問題: {query}")
        print("-" * 50)
        print(f"💡 回答:\n{response['result']}")
        print("="*50)

if __name__ == "__main__":
    main()