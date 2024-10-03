import streamlit as st
import pandas as pd
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import tempfile
import os

# Set the page configuration (optional)
st.set_page_config(page_title="LangChain FAISS 색인 내용", layout="wide")

st.title("LangChain FAISS 색인 내용")

# Add a file uploader for the individual FAISS index files
uploaded_files = st.file_uploader(
    "'index.faiss'와 'index.pkl' 파일을 업로드하세요",
    type=["faiss", "pkl"],
    accept_multiple_files=True
)

if uploaded_files is not None and len(uploaded_files) > 0:
    # Create a temporary directory to save the uploaded files
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Initialize variables to store the paths of the uploaded files
        index_faiss_path = None
        index_pkl_path = None

        # Process the uploaded files
        for uploaded_file in uploaded_files:
            # Save each file to the temporary directory
            file_path = os.path.join(tmpdirname, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Identify the file based on its name
            if uploaded_file.name == 'index.faiss':
                index_faiss_path = file_path
            elif uploaded_file.name == 'index.pkl':
                index_pkl_path = file_path

        # Check if both required files have been uploaded
        if index_faiss_path is None or index_pkl_path is None:
            st.error("'index.faiss'와 'index.pkl' 파일을 모두 업로드해주세요.")
            st.stop()

        # Now, load the FAISS index from the temporary directory
        # Since the files are saved in tmpdirname, we can use that as the base path
        try:
            vectorstore = FAISS.load_local(
                tmpdirname,
                embeddings=OpenAIEmbeddings(model="text-embedding-ada-002"),
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            st.error(f"FAISS 색인 로드 실패: {e}")
            st.stop()

        # Proceed with processing and displaying the data
        # Get the total number of vectors
        n_vectors = vectorstore.index.ntotal

        # Initialize lists to hold data
        texts = []
        metadatas = []
        embeddings_str = []

        # Function to convert embeddings to string with ellipsis, showing only the first 100 numbers
        def embedding_to_str(embedding):
            truncated_embedding = embedding[:100]  # Take the first 100 numbers
            embedding_str = ", ".join("{:.3f}".format(num) for num in truncated_embedding)
            return "[{}...]".format(embedding_str)

        # Iterate over the indices
        for i in range(n_vectors):
            # Get the document ID
            doc_id = vectorstore.index_to_docstore_id[i]

            # Retrieve the document
            doc = vectorstore.docstore.search(doc_id)

            # Append text and metadata
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)

            # Reconstruct the embedding vector
            embedding_vector = vectorstore.index.reconstruct(i)

            # Convert embedding to string with ellipsis
            embeddings_str.append(embedding_to_str(embedding_vector))

        # Create the DataFrame
        df = pd.DataFrame({
            'text': texts,
            'metadata': metadatas,
            'embeddings': embeddings_str
        })

        # Function to convert DataFrame to HTML with code blocks in 'metadata' column
        def df_to_html_with_code(df):
            html = '''
            <html>
            <head>
                <style>
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        table-layout: fixed;
                        word-wrap: break-word;
                    }
                    th, td {
                        text-align: left;
                        vertical-align: top;
                        padding: 8px;
                        border-bottom: 1px solid #ddd;
                        word-wrap: break-word;
                    }
                    th {
                        background-color: #f2f2f2;
                    }
                    pre {
                        background-color: #f0f0f0;
                        padding: 8px;
                        margin: 0;
                        white-space: pre-wrap; /* Wrap long lines */
                        word-wrap: break-word; /* Break long words */
                        font-size: 0.9em;
                    }
                    code {
                        font-family: Consolas, 'Courier New', monospace;
                    }
                    .cell-content {
                        max-height: 200px;
                        overflow: auto;
                    }
                </style>
            </head>
            <body>
                <table>
            '''
            # Add table header
            html += '<tr>'
            for column in df.columns:
                html += f'<th>{column}</th>'
            html += '</tr>'
            # Add table rows
            for _, row in df.iterrows():
                html += '<tr>'
                for column in df.columns:
                    cell_value = row[column]
                    if column == 'metadata':
                        # Convert metadata to JSON string with indentation
                        import json
                        metadata_str = json.dumps(cell_value, indent=2, ensure_ascii=False)
                        # Wrap in code block
                        cell_html = f'''
                        <div class="cell-content">
                            <pre><code>{metadata_str}</code></pre>
                        </div>
                        '''
                    else:
                        cell_html = f'''
                        <div class="cell-content">
                            {cell_value}
                        </div>
                        '''
                    html += f'<td>{cell_html}</td>'
                html += '</tr>'
            html += '''
                </table>
            </body>
            </html>
            '''
            return html

        # Convert DataFrame to HTML
        html_table = df_to_html_with_code(df)

        # Display the HTML table using st.components.v1.html()
        st.components.v1.html(html_table, height=800, scrolling=True)
    # Temporary directory and its contents are cleaned up here
else:
    st.info("FAISS 색인 내용을 보려면 'index.faiss'와 'index.pkl' 파일을 모두 업로드해주세요.")
