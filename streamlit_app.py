import streamlit as st
import os
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import xmltodict

# XMLファイルから特定の要素のテキスト内容を取得する関数
def get_xml_text_content(xml_file, element_name='HASSEIJI_JOKYO_TXT'):
    try:
        with open(xml_file, 'r', encoding='utf-8') as file:
            doc = xmltodict.parse(file.read())
            # XMLの構造によってこのパスは調整する必要があります
            text_content = doc['root'][element_name]
            return text_content
    except Exception as e:
        st.error(f"Error processing file {xml_file}: {e}")
        return ""

# 検索実行関数
def perform_search(query_text, files_data, search_type):
    texts = [data['text'] for data in files_data]
    if search_type == "Similarity Search":
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([query_text] + texts)
        cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
        results = [(files_data[i-1]['name'], cosine_similarities[i]) for i in range(1, len(files_data) + 1)]
    else:  # Content Search
        results = [(file_data['name'], query_text.lower() in file_data['text'].lower()) for file_data in files_data]
        results = [result for result in results if result[1]]
    
    # スコアでソート
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]

# Streamlit UI
def main():
    st.title("XML Document Search Tool")
    
    # 入力テキストボックス
    query_text = st.text_area("Enter text for search:", height=300, key="query_text")
    
    # 検索ボタン
    search_type = st.radio("Select search type:", ("Similarity Search", "Content Search"))
    
    # XMLデータの読み込み
    xml_directory = './xml'
    files_data = []
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            text_content = get_xml_text_content(file_path)
            files_data.append({'name': filename, 'text': text_content})
    
    if st.button("Search"):
        results = perform_search(query_text, files_data, search_type)
        option = st.selectbox('Select an XML file:', [result[0] for result in results])
        
        # 選択されたXMLファイルの内容を表示
        selected_file_path = os.path.join(xml_directory, option)
        selected_file_content = get_xml_text_content(selected_file_path)
        st.text_area("File Content:", value=selected_file_content, height=300, key="file_content")

if __name__ == "__main__":
    main()
