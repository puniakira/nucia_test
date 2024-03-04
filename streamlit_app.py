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

# 類似度検索機能
def search_similar_documents(query_text, xml_files, xml_texts):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query_text] + xml_texts)
    cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
    similarity_scores = list(enumerate(cosine_similarities[1:], start=1))

    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
    return [(xml_files[idx-1], score) for idx, score in sorted_scores]

# 部分一致検索機能
def search_documents_by_content(keyword, xml_files, xml_texts):
    return [file for file, text in zip(xml_files, xml_texts) if keyword.lower() in text.lower()]

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

if st.button("Search"):
    if search_type == "Similarity Search":
        results = search_similar_documents(query_text, xml_files, xml_texts)
    else:
        results = search_documents_by_content(query_text, xml_files, xml_texts)

    for filename, score in results:
        st.write(f"{filename} - Score: {score:.4f}")

selected_file = st.selectbox("Select an XML file to display content:", xml_files)

if selected_file:
    content = get_xml_text_content(os.path.join(xml_directory, selected_file))
    st.text_area("Content:", value=content, height=300)
