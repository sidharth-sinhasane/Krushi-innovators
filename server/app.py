from flask import Flask, request, jsonify
from ultralytics import YOLO
import os

model = YOLO('best.pt')

app = Flask(__name__)

# Specify the upload folder
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set the maximum file size (in bytes) - 5MB in this example
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    
# Define a route to receive POST requests with image file uploads
@app.route('/api/upload', methods=['POST'])
def upload_file():
    # Check if the POST request contains a file
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    files= request.files.getlist('file')
    no_img=len(file)
    avg_confi=0
    lab_prev=""
    check_flag=False
    for file in files:
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
    # Check if the file is allowed based on its extension
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Unsupported file format'}), 400

    # Save the uploaded file to the upload folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
    
        results = model.predict(file_path)
        result = results[0]
        boxes = result.boxes
        cls_np = boxes.cls.cpu().numpy().astype(int)
        conf_np = boxes.conf.cpu().numpy()
        prediction = None
        
        if len(cls_np)>0:
            label = cls_np[0]
            confidence = conf_np[0]
            avg_confi+=int(confidence*100)
            prediction = {
            'name':model.names[label]
            }
        if lab_prev=="":
            lab_prev=model.names[label]
            continue
        if lab_prev != model.names[label]:
            check_flag=True
            break
        lab_prev=model.names[label]
    if check_flag==True:
         prediction['confidence']=0
    else:
        prediction['confidence']=avg_confi/no_img
    print(prediction)

    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path, 'prediction':prediction}), 200

@app.route('/',methods=['GET'])
def welcome():
    return jsonify({'msg':'Hello World'})

if __name__ == '__main__':
    app.run()