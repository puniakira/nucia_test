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
        text_content = root.find(f'.//{element_name}').text
        return text_content
    except Exception as e:
        st.error(f"Error processing file {xml_file}: {e}")
        return ""

# Streamlit UI
st.title("XML Document Search")

query_text = st.text_area("Enter text for search:", height=150)
search_type = st.radio("Select search type:", ("Similarity Search", "Content Search"))

xml_directory = './xml'
xml_files = []
xml_texts = []

for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_directory, filename)
        xml_text = get_xml_text_content(file_path)
        if xml_text:
            xml_texts.append(xml_text)
            xml_files.append(filename)

# 検索実行
if st.button("Search"):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query_text] + xml_texts)
    cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
    scores = list(enumerate(cosine_similarities[1:], start=1))

    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]
    sorted_files = [(xml_files[idx-1], score) for idx, score in sorted_scores]

    # 検索結果のファイル名とスコアを表示
   
