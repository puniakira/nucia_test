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
                xml_texts.append(xml_text)
                xml_files.append(filename)

    results = []
    if xml_texts:
        if search_type == 'similarity':
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([keyword] + xml_texts)
            cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
            similarity_scores = list(enumerate(cosine_similarities[1:], start=1))
            sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
            for idx, score in sorted_scores:
                results.append((xml_files[idx-1], score))
        elif search_type == 'content':
            for idx, text in enumerate(xml_texts, start=1):
                if keyword.lower() in text.lower():
                    results.append((xml_files[idx-1], "Contains keyword"))

    return results

# Streamlitアプリケーションのメイン関数
def main():
    st.title("XML Search and Similarity Tool")
    
    # キーワード入力（大きなテキストボックス）
    keyword = st.text_area("Enter keyword for search or similarity search:", height=150)
    
    col1, col2 = st.columns(2)
    with col1:
        # 類似度検索ボタン
        if st.button("Search Similar Documents"):
            if keyword:
                results = search_similar_documents(keyword, search_type='similarity')
                display_results(results)
            else:
                st.error("Please enter a keyword.")
    with col2:
        # 部分一致検索ボタン
        if st.button("Search Documents by Content"):
            if keyword:
                results = search_similar_documents(keyword, search_type='content')
                display_results(results)
            else:
                st.error("Please enter a keyword.")

def display_results(results):
    selected_file = st.selectbox("Select a file to view content", [result[0] for result in results])
    file_path = os.path.join('./xml', selected_file)
    xml_text = get_xml_text_content(file_path)
    st.text_area("XML Content:", value=xml_text, height=300)

if __name__ == "__main__":
    main()
