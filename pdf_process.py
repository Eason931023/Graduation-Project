def process_pdf_data(file_path: str, cache_path: str) -> List[Document]:
    """
    處理 PDF 檔案流程：載入 -> 結構化清理/表格轉 Markdown -> 產出 JSON 緩存。
    
    注意：UnstructuredFileLoader 會嘗試自動偵測表格並轉為 Markdown，這裡主要利用其功能。
    """
    
    # 1. 載入與結構化清理
    print(f"--- 開始處理 PDF: {file_path} ---")
    
    # 這裡使用 UnstructuredFileLoader，它會嘗試進行版面分析和表格轉 Markdown
    # 這是處理複雜 PDF 的推薦方法
    try:
        loader = UnstructuredFileLoader(
            file_path, 
            mode="elements", # 以元素（區塊、段落、表格）模式載入
            strategy="hi_res" # 使用高解析度策略 (需額外安裝 tesseract 或其他 OCR)
        )
        # LangChain 的 Document 格式
        documents: List[Document] = loader.load() 
    except Exception as e:
        print(f"❌ 載入 PDF 發生錯誤: {e}")
        return []

    processed_data = []
    for doc in documents:
        # 2. 結構化清理與 Metadata 調整
        
        # 確保 page_content 是字串且非空白
        content = doc.page_content.strip()
        if not content:
            continue
            
        # 統一 Metadata
        metadata = doc.metadata.copy()
        metadata['source_file'] = os.path.basename(file_path)
        metadata['data_type'] = 'PDF_Structural'
        
        # 移除 Unstructured 內建但不需要的欄位
        metadata.pop('filename', None)
        metadata.pop('file_directory', None)
        
        processed_data.append({
            "page_content": content,
            "metadata": metadata
        })
        
    # 3. 產出 JSON 緩存檔
    save_to_json_cache(processed_data, cache_path)
    
    # 4. 載入為 Document 物件（為後續 Embedding 做準備）
    return load_from_json_cache(cache_path)