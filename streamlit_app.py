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

# XMLファイルとの類似度検索
def search_similar_documents(keyword, xml_directory='./xml', search_type='similarity'):
    xml_texts = []
    xml_files = []

    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            xml_text = get_xml_text_content(file_path)
            if xml_text:
                if search_type == 'content' and keyword.lower() in xml_text.lower():
                    xml_texts.append(xml_text)
                    xml_files.append(filename)
                elif search_type == 'similarity':
                    xml_texts.append(xml_text)
                    xml_files.append(filename)

    results = []
    if xml_texts and search_type == 'similarity':
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([keyword] + xml_texts)
        cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
        similarity_scores = list(enumerate(cosine_similarities[1:], start=1))
        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
        results = [(xml_files[idx-1], score) for idx, score in sorted_scores]
    elif xml_texts and search_type == 'content':
        results = [(file, 0) for file in xml_files]

    return results

# Streamlitアプリケーションのメイン関数
def main():
    st.title("XML Search and Similarity Tool")

    # キーワード入力
    keyword = st.text_area("Enter keyword for search or similarity search:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Similar Documents"):
            results = search_similar_documents(keyword, search_type='similarity')
            for filename, score in results:
                st.write(f"{filename}: Score = {score:.4f}")

    with col2:
        if st.button("Search Documents by Content"):
            results = search_similar_documents(keyword, search_type='content')
            for filename, _ in results:
                st.write(f"{filename}")

    # 検索結果の一覧表示とHASSEIJI_JOKYO_TXTの表示
    if results:
        selected_file = st.selectbox("Select a file to view its HASSEIJI_JOKYO_TXT:", [result[0] for result in results])
        if st.button("Show HASSEIJI_JOKYO_TXT"):
            file_path = os.path.join('./xml', selected_file)
            xml_text = get_xml_text_content(file_path)
            st.text_area("HASSEIJI_JOKYO_TXT:", value=xml_text, height=300)

if __name__ == "__main__":
    main()
