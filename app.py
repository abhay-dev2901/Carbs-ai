
import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from predict import predict_food

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Run prediction
            results = predict_food(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify(results)
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'Carbs-AI Backend Running'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
