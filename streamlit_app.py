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

# 検索実行関数
def perform_search():
    if search_type == "Similarity Search":
        st.session_state.results = search_similar_documents(st.session_state.query_text, xml_files, xml_texts)
    else:
        st.session_state.results = []  # 部分一致検索の結果をここに設定する

# 類似度検索機能
def search_similar_documents(query_text, xml_files, xml_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query_text] + xml_texts)
    cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
    similarity_scores = list(enumerate(cosine_similarities[1:], start=1))

    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
    return [(xml_files[idx-1], score) for idx, score in sorted_scores]

# Streamlit UI
st.title("XML Document Search")

if 'query_text' not in st.session_state:
    st.session_state.query_text = ''
if 'results' not in st.session_state:
    st.session_state.results = []

st.session_state.query_text = st.text_area("Enter text for search:", height=150, key='query_text')
search_type = st.radio("Select search type:", ("Similarity Search", "Content Search"))

xml_directory = './xml'
xml_files = []
xml_texts = []

# XMLファイルを読み込む
for filename in os.listdir(xml_directory):
    if filename.endswith('.xml'):
        file_path = os.path.join(xml_directory, filename)
        xml_text = get_xml_text_content(file_path)
        if xml_text:
            xml_texts.append(xml_text)
            xml_files.append(filename)

if st.button("Search"):
    perform_search()

result_files = [f"{filename} - Score: {score:.4f}" for filename, score in st.session_state.results]
selected_result = st.selectbox("Select a result:", result_files, key='selected_result')

if selected_result:
    selected_file = selected_result.split(" - ")[0]
    content = get_xml_text_content(os.path.join(xml_directory, selected_file))
    st.text_area("Content:", value=content, height=300, key='content_area')
