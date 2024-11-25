from flask import Flask, render_template, request, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from transformers import BertTokenizer, BertModel
from PIL import Image
from utils import preprocess_answer, save_uploaded_file

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'

# Define the VQA Model class
class VQAModel1(nn.Module):
    def __init__(self, num_answers):
        super(VQAModel1, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(2048 + 768, 1024)
        self.fc2 = nn.Linear(1024, num_answers)
        self.dropout = nn.Dropout(0.5)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.cnn(images)
        outputs = self.bert(input_ids, attention_mask)
        question_features = outputs.last_hidden_state[:, 0, :]
        combined_features = torch.cat((image_features, question_features), dim=1)
        x = self.fc1(combined_features)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VQAModel1(num_answers=582)
checkpoint = torch.load("./model/best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# Tokenizer and transforms
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load answer space
with open("./model/answer_space.txt") as f:
    answer_space = f.read().splitlines()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if image is uploaded
        if 'image' not in request.files or not request.files['image']:
            return redirect(request.url)
        
        # Save the uploaded file
        file = request.files['image']
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'])

        # Process question
        question = request.form.get('question')
        if not question:
            return redirect(request.url)

        # Preprocess image
        image = Image.open(file_path).convert("RGB")
        image_tensor = image_transforms(image).unsqueeze(0).to(device)

        # Tokenize question
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=50)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor, input_ids, attention_mask)
            _, predicted_idx = torch.max(outputs, dim=1)
            predicted_answer = preprocess_answer(answer_space[predicted_idx.item()])

        return render_template('index.html', prediction=predicted_answer, image_url=file_path)
    return render_template('index.html', prediction=None)

if __name__ == '__main__':
    app.run(debug=True)
