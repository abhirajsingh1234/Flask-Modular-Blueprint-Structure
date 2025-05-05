import os
from langchain_community.document_loaders import PyPDFLoader


def extract_data(folder_path,file):
    """
    Extracts content from a PDF file.

    Args:
        folder_path (str): Path to the folder containing the PDF file.

    Returns:
        str: Extracted content from the PDF file.
    """
    data = ''
    
    

    if file.lower().endswith(".pdf"):

        print()

        file_path = os.path.join(folder_path, file)

        loader = PyPDFLoader(file_path)

        documents = loader.load()

        for document in documents:

            data += document.page_content + '\n'



    return data