import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from pathlib import Path

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load full model directly (not just state_dict)
model = torch.load("C:\DeepFake_Defenders\final_video_model.pth", map_location=device)
model.to(device)
model.eval()

# Define preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Adjust if your model expects different normalization
])

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = cap.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
        if len(frames) == num_frames:
            break

    cap.release()
    return torch.stack(frames)  # Shape: [num_frames, 3, 224, 224]

def predict_video(video_path):
    print(f"\n[Test] Processing: {video_path}")
    frames = extract_frames(video_path).unsqueeze(0).to(device)  # Shape: [1, T, 3, 224, 224]
    with torch.no_grad():
        outputs = model(frames)
        probs = torch.sigmoid(outputs).cpu().numpy()[0][0]
        label = "DeepFake" if probs >= 0.5 else "Real"
        print(f"[Test] Prediction: {label} (Confidence: {probs:.4f})")

# Example test
if __name__ == "__main__":
    test_video = r"C:\DeepFake_Defenders\dataset\manipulated_sequences\Face2Face\c23\videos\060_088.mp4"
    predict_video(test_video)
