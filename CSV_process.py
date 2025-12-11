import pandas as pd

def process_csv_data(file_path: str, cache_path: str) -> List[Document]:
    """
    處理 CSV 檔案流程：讀取 -> 轉化為語意文本塊 (行轉 Markdown) -> 產出 JSON 緩存。
    """
    print(f"--- 開始處理 CSV: {file_path} ---")
    
    # 1. 讀取 CSV 檔案
    try:
        # 嘗試用 UTF-8 讀取，若失敗可嘗試 'big5' 或 'gbk'
        df = pd.read_csv(file_path, encoding='utf-8').fillna('')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='big5').fillna('')
        except Exception as e:
            print(f"❌ 讀取 CSV 檔案失敗: {e}")
            return []

    headers = list(df.columns)
    markdown_header = "| " + " | ".join(headers) + " |"
    markdown_separator = "|---" * len(headers) + "|"

    processed_data = []
    source_filename = os.path.basename(file_path)

    # 2. 轉化為語意文本塊 (行轉 Markdown 策略)
    for index, row in df.iterrows():
        # 2.1 產生 Markdown 格式的內容
        row_values = [str(val) for val in row.values]
        markdown_row = "| " + " | ".join(row_values) + " |"
        
        chunk_text = (
            f"以下是原始資料的一筆結構化紀錄 (行 ID: {index+1})，已轉換為 Markdown 表格:\n"
            f"{markdown_header}\n{markdown_separator}\n{markdown_row}"
        )
        
        # 2.2 建立 Document Chunk 及其 Metadata
        metadata = {
            "source_file": source_filename,
            "data_type": "CSV_Record",
            "row_index": index,
            # 可將關鍵欄位的值提取出來作為可過濾的 Metadata
            "key_field_1": str(row[headers[0]]) if headers else None 
        }

        processed_data.append({
            "page_content": chunk_text,
            "metadata": metadata
        })
        
    # 3. 產出 JSON 緩存檔
    save_to_json_cache(processed_data, cache_path)
    
    # 4. 載入為 Document 物件
    return load_from_json_cache(cache_path)