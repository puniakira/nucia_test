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
def search_similar_documents(keyword, xml_directory='./xml', element_name='HASSEIJI_JOKYO_TXT'):
    xml_texts = []
    xml_files = []

    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            xml_text = get_xml_text_content(file_path, element_name)
            if xml_text:
                xml_texts.append(xml_text)
                xml_files.append(filename)

    results = []
    if xml_texts:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([keyword] + xml_texts)
        cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
        similarity_scores = list(enumerate(cosine_similarities[1:], start=1))

        sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
        for idx, score in sorted_scores:
            results.append((xml_files[idx-1], score, get_xml_text_content(os.path.join(xml_directory, xml_files[idx-1]), element_name)))

    return results

# 部分一致検索
def search_documents_by_content(keyword, xml_directory='./xml', element_name='HASSEIJI_JOKYO_TXT'):
    results = []
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            file_path = os.path.join(xml_directory, filename)
            xml_text = get_xml_text_content(file_path, element_name)
            if keyword.lower() in xml_text.lower():
                results.append((filename, xml_text))

    return results[:10]

# Streamlitアプリケーションのメイン関数
def main():
    st.title("XML Search and Similarity Tool")

    # キーワード入力
    keyword = st.text_area("Enter keyword for search or similarity search:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Search Similar Documents"):
            if keyword:
                results = search_similar_documents(keyword)
                for filename, score, _ in results:
                    st.write(f"{filename}: Score = {score:.4f}")
            else:
                st.error("Please enter a keyword.")
    
    with col2:
        if st.button("Search Documents by Content"):
            if keyword:
                results = search_documents_by_content(keyword)
                for filename, _ in results:
                    st.write(f"{filename}")
            else:
                st.error("Please enter a keyword.")
    
    selected_filename = st.selectbox("Select a file to show its HASSEIJI_JOKYO_TXT content:", [result[0] for result in results], index=0)
    
    if st.button("Show Selected Document Content"):
        for filename, _, content in results:
            if filename == selected_filename:
                st.text_area("HASSEIJI_JOKYO_TXT:", value=content, height=300)

if __name__ == "__main__":
    main()
