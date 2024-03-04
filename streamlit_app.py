import tkinter as tk
from tkinter import ttk
import os
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# XMLファイルから特定の要素のテキスト内容を取得する関数
def get_xml_text_content(xml_file, element_name):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        text_content = root.find(f'.//{element_name}').text
        return text_content
    except Exception as e:
        print(f"Error processing file {xml_file}: {e}")
        return ""

class XMLSearchApp:
    def __init__(self, master):
        self.master = master
        master.title("XML Search and Similarity Tool")

        # キーワード入力用のラベルとテキストボックス
        self.label = tk.Label(master, text="Enter keyword for search or similarity search:")
        self.label.pack()

        self.entry = tk.Entry(master, width=50)
        self.entry.pack()

        # 類似度検索ボタン
        self.search_similarity_button = tk.Button(master, text="Search Similar Documents", command=self.search_similar_documents)
        self.search_similarity_button.pack()

        # 部分一致検索ボタン
        self.search_button = tk.Button(master, text="Search Documents by Content", command=self.search_documents_by_content)
        self.search_button.pack()

        # 検索結果表示用のリストボックス
        self.listbox = tk.Listbox(master, width=100, height=10)
        self.listbox.pack()
        self.listbox.bind('<<ListboxSelect>>', self.show_xml_content)

        # XMLの内容を表示するテキストエリア
        self.text_area = tk.Text(master, height=15, width=100)
        self.text_area.pack()

    def search_similar_documents(self):
        # 類似度検索機能
        keyword = self.entry.get()
        self.listbox.delete(0, tk.END)
        xml_directory = './xml'
        xml_texts = []
        xml_files = []

        for filename in os.listdir(xml_directory):
            if filename.endswith('.xml'):
                file_path = os.path.join(xml_directory, filename)
                xml_text = get_xml_text_content(file_path, "HASSEIJI_JOKYO_TXT")
                if xml_text:
                    xml_texts.append(xml_text)
                    xml_files.append(filename)

        if xml_texts:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([keyword] + xml_texts)
            cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
            similarity_scores = list(enumerate(cosine_similarities[1:], start=1))

            sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[:10]
            for idx, score in sorted_scores:
                self.listbox.insert(tk.END, f"{xml_files[idx-1]} (Score: {score:.4f})")

    def search_documents_by_content(self):
        # 部分一致検索機能
        keyword = self.entry.get().lower()
        self.listbox.delete(0, tk.END)
        xml_directory = './xml'

        for filename in os.listdir(xml_directory):
            if filename.endswith('.xml'):
                file_path = os.path.join(xml_directory, filename)
                xml_text = get_xml_text_content(file_path, "HASSEIJI_JOKYO_TXT")
                if keyword in xml_text.lower():
                    self.listbox.insert(tk.END, filename)

    def show_xml_content(self, event):
        # 選択されたXMLの内容を表示
        selection = event.widget.curselection()
        if selection:
            index = selection[0]
            filename = event.widget.get(index).split(" ")[0]
            file_path = os.path.join('./xml', filename)

            xml_text = get_xml_text_content(file_path, "HASSEIJI_JOKYO_TXT")
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END, xml_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = XMLSearchApp(root)
    root.mainloop()
