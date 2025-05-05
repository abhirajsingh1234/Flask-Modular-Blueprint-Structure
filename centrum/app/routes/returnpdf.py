# from flask import Blueprint, request, jsonify
# import os
# import base64

# returnpdf_bp = Blueprint('returnpdf', __name__)
# path="data"

# @returnpdf_bp.route('/', methods=['GET'])
# def home():
#     return jsonify({"message":"getting started"}), 200

# from flask import Blueprint, request, jsonify
# import os
# import base64

# returnpdf_bp = Blueprint('returnpdf', __name__)
# DATA_DIR = "data"

# @returnpdf_bp.route('/', methods=['GET'])
# def home():
#     return jsonify({"message": "getting started"}), 200

# @returnpdf_bp.route('/', methods=['POST'])
# def returnpdf():
#     data = request.get_json(force=True)
#     pdf_name = data.get("pdf_name")
#     if not pdf_name:
#         return jsonify({"error": "`pdf_name` is required"}), 400

#     try:
#         # list files in your data directory
#         dir_list = os.listdir(DATA_DIR)
#         if pdf_name not in dir_list:
#             return jsonify({"error": f"File '{pdf_name}' not found"}), 404

#         file_path = os.path.join(DATA_DIR, pdf_name)
#         with open(file_path, "rb") as pdf_file:
#             encoded_bytes = base64.b64encode(pdf_file.read())
#             encoded_string = encoded_bytes.decode('utf-8')
    
#     except Exception as e:
#         # catch everything else and send the exception message
#         return jsonify({"error": str(e)}), 500

#     return jsonify({"encoded_string": encoded_string}), 200

from flask import Blueprint, request, jsonify
import os
import base64
from functools import lru_cache

# Create Blueprint
returnpdf_bp = Blueprint('returnpdf', __name__)

# Constants
DATA_DIR = os.getenv("DATA_DIR", "data")

# Cache for frequently accessed files to improve performance
@lru_cache(maxsize=20)  # Adjust cache size based on your needs
def get_encoded_pdf(filename):
    """Get base64 encoded PDF with caching for improved performance"""
    file_path = os.path.join(DATA_DIR, filename)
    
    try:
        with open(file_path, "rb") as pdf_file:
            encoded_bytes = base64.b64encode(pdf_file.read())
            return encoded_bytes.decode('utf-8')
    except Exception as e:
        # Return None and handle the error in the route
        return None

@returnpdf_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "PDF Retrieval API is ready"}), 200

@returnpdf_bp.route('/', methods=['POST'])
def returnpdf():
    # Parse request data
    data = request.get_json(force=True)
    pdf_name = data.get("pdf_name")
    
    # Validate request
    if not pdf_name:
        return jsonify({"error": "`pdf_name` is required"}), 400

    # Validate file exists
    if not os.path.exists(os.path.join(DATA_DIR, pdf_name)):
        return jsonify({"error": f"File '{pdf_name}' not found"}), 404
    
    # Get cached or new encoded PDF
    encoded_string = get_encoded_pdf(pdf_name)
    
    if encoded_string is None:
        return jsonify({"error": f"Failed to read file '{pdf_name}'"}), 500
    
    return jsonify({"encoded_string": encoded_string}), 200

@returnpdf_bp.route('/list', methods=['GET'])
def list_pdfs():
    """New endpoint to list available PDFs"""
    try:
        pdf_files = [
            file for file in os.listdir(DATA_DIR) 
            if file.lower().endswith(".pdf")
        ]
        return jsonify({"pdf_files": pdf_files}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
