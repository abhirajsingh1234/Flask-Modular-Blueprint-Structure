from flask import Blueprint, request, jsonify
import os
import base64

returnpdf_bp = Blueprint('returnpdf', __name__)
path="data"

@returnpdf_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message":"getting started"}), 200

from flask import Blueprint, request, jsonify
import os
import base64

returnpdf_bp = Blueprint('returnpdf', __name__)
DATA_DIR = "data"

@returnpdf_bp.route('/', methods=['GET'])
def home():
    return jsonify({"message": "getting started"}), 200

@returnpdf_bp.route('/', methods=['POST'])
def returnpdf():
    data = request.get_json(force=True)
    pdf_name = data.get("pdf_name")
    if not pdf_name:
        return jsonify({"error": "`pdf_name` is required"}), 400

    try:
        # list files in your data directory
        dir_list = os.listdir(DATA_DIR)
        if pdf_name not in dir_list:
            return jsonify({"error": f"File '{pdf_name}' not found"}), 404

        file_path = os.path.join(DATA_DIR, pdf_name)
        with open(file_path, "rb") as pdf_file:
            encoded_bytes = base64.b64encode(pdf_file.read())
            encoded_string = encoded_bytes.decode('utf-8')
    
    except Exception as e:
        # catch everything else and send the exception message
        return jsonify({"error": str(e)}), 500

    return jsonify({"encoded_string": encoded_string}), 200
