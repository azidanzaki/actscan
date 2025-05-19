from flask import Flask, request, render_template, redirect, url_for
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# 8 label aktivitas
labels = ['Menelpon', 'Minum', 'Makan', 'Berkelahi', 'Berpelukan', 'Berlari', 'Duduk', 'Tidur']

# Model klasifikasi 8 kelas
class CustomResNet18(nn.Module):
    def __init__(self):
        super(CustomResNet18, self).__init__()
        self.cnn = models.resnet18(weights=None)
        num_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Linear(num_features, 8)

    def forward(self, x):
        return self.cnn(x)

# Load model
model = CustomResNet18()
model.load_state_dict(torch.load('kelompok10.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ✅ Default route membuka halaman dashboard
@app.route('/')
def default():
    return render_template('dashboard.html')

@app.route('/dashbard', methods=['GET', 'POST'])
def dashboard():
    return render_template('dashboard.html')

# ✅ Route untuk deteksi gambar
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return 'No image uploaded', 400

        file = request.files['image']
        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocessing dan prediksi
        image = Image.open(filepath).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(image_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_label = labels[predicted_idx]

        return render_template('result.html', prediction=predicted_label, filename=filename)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5000)
