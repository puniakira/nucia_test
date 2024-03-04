import streamlit as st
import os
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# XMLファイルから特定の要素のテキスト内容を取得する関数
def get_xml_text_content(xml_file, element_name='HASSEIJI_JOKYO_TXT'):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text_content = root.find(f'.//{element_name}')
        return text_content.text if text_content is not None else "No content"
    except Exception as e:
        st.error(f"Error processing file {xml_file}: {e}")
        return "Error"

# 指定されたフォルダ内のXMLファイルを読み込む
def load_xml_files(folder_path='./xml'):
    xml_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xml')]
    return xml_files

# XMLファイルとの類似度または部分一致を検索
def search_documents(keyword, xml_files, search_type='similarity'):
    texts, filenames = [], []
    for xml_file in xml_files:
        text = get_xml_text_content(xml_file)
        if search_type == 'content' and keyword.lower() in text.lower():
            texts.append(text)
            filenames.append(os.path.basename(xml_file))
        elif search_type == 'similarity':
            texts.append(text)
            filenames.append(os.path.basename(xml_file))
    
    if search_type == 'similarity' and texts:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([keyword] + texts)
        cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
        scores = [(filenames[i-1], cosine_similarities[i]) for i in range(1, len(cosine_similarities))]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
        return sorted_scores
    elif search_type == 'content':
        return [(filename, 0) for filename in filenames]  # スコアは使用しない
    else:
        return []

# メイン関数
def main():
    st.title("XML Search and Similarity Tool")

    # キーワード入力
    keyword = st.text_area("Enter keyword for search or similarity search:", height=150)
    
    # 検索結果表示用のリスト
    results_list = st.empty()

    # 類似度検索ボタン
    if st.button("Search Similar Documents"):
        if keyword:
            xml_files = load_xml_files()
            results = search_documents(keyword, xml_files, search_type='similarity')
            results_list = [f"{filename}: Score = {score:.4f}" for filename, score in results]
            selected = st.selectbox("Select a document:", results_list)
        else:
            st.error("Please enter a keyword.")

    # 部分一致検索ボタン
    if st.button("Search Documents by Content"):
        if keyword:
            xml_files = load_xml_files()
            results = search_documents(keyword, xml_files, search_type='content')
            results_list = [filename for filename, _ in results]
            selected = st.selectbox("Select a document:", results_list)
        else:
            st.error("Please enter a keyword.")

    # 選択したリストの内容を表示するボタン
    show_content_button = st.button("Show Selected Document Content")
    if show_content_button and results_list:
        selected_filename = selected.split(":")[0].strip()
        selected_file_path = os.path.join('./xml', selected_filename)
        content = get_xml_text_content(selected_file_path)
        st.text_area("Document Content:", value=content, height=300)

if __name__ == "__main__":
    main()
