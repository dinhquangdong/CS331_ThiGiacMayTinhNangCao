import streamlit as st
import cv2
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms


device = "cuda" if torch.cuda.is_available() else "cpu"

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=128*26*26, out_features=128)
        
        # Connecting CNN outputs with Fully Connected layers for classification
        self.fc_class = nn.Linear(in_features=128, out_features=1)
        
        # Connecting CNN outputs with Fully Connected layers for bounding box
        self.fc_box = nn.Linear(in_features=128, out_features=4)
        
    
    def forward(self, x):
        x = self.max_pool2d(torch.relu(self.conv1(x)))
        x = self.max_pool2d(torch.relu(self.conv2(x)))
        x = self.max_pool2d(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        
        # classification with sigmoid
        class_x = self.fc_class(x)
        class_x = torch.sigmoid(class_x)
        
        # bounding box with no activation function
        box_x = self.fc_box(x)
        
        return [class_x, box_x]
    

@st.cache_resource
def load_model(path_to_model):
    model = MyModel()
    model.load_state_dict(torch.load(path_to_model))
    model.to(device)
    return model


def show_result(model, test_image):
    test_image = cv2.resize(test_image, (224, 224))
    show_im = test_image.copy()
    
    test_image = np.float32(test_image) / 255.0
    transform = transforms.ToTensor()
    test_image_tensor = transform(test_image)
    test_image_tensor = test_image_tensor.unsqueeze(0)
    test_image_tensor = test_image_tensor.to(device)

    model.eval()
    with torch.no_grad():
        [class_pred, box_pred] = model(test_image_tensor)


    label = "fire hydrant" if class_pred.item()<=0.5 else "teapot"

    x1 = box_pred[0][0]
    y1 = box_pred[0][1]
    x2 = box_pred[0][2]
    y2 = box_pred[0][3]

    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))

    show_im = cv2.rectangle(show_im, pt1, pt2, (0, 255, 0), 2)
    return label, show_im
    

# Load model
model = load_model("Models/trained_model.pth")

st.title("Demo")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    np_array = np.frombuffer(file_bytes, np.uint8)
    # Đọc hình ảnh bằng OpenCV
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label, im = show_result(model=model, test_image=image)
    
    st.subheader(label)
    st.image(im)
    