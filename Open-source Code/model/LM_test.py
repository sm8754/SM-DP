import os
import clip
import torch
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
weights_dict = torch.load(".pth", map_location='cuda')
model.load_state_dict(weights_dict, strict=True)
image_folder = 'path_to_valid_dataset'

classes = ['Tumor', 'Tissue', 'Background', 'ROI']
correct_predictions = 0
total_images = 0
true_labels = []
predicted_labels = []

for class_name in classes:
    class_folder = os.path.join(image_folder, class_name)
    for subdir in os.listdir(class_folder):
        subdir_path = os.path.join(class_folder, subdir)
        for image_file in tqdm(os.listdir(subdir_path), desc=f"Processing {class_name} images", unit="image"):
            try:
                if image_file.endswith('.png'):
                    total_images += 1
                    image_path = os.path.join(subdir_path, image_file)
                    image = Image.open(image_path).convert("RGB")
                    image_input = preprocess(image).unsqueeze(0).to(device)
                    text_inputs = torch.cat([clip.tokenize(f"{c} image of the liver.") for c in classes]).to(device)
                    with torch.no_grad():
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_inputs)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    predicted_class = classes[similarity[0].argmax().item()]
                    true_labels.append(class_name)
                    predicted_labels.append(predicted_class)
                    if predicted_class == class_name:
                        correct_predictions += 1
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
plt.figure(figsize=(8, 8))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
accuracy = correct_predictions / total_images * 100
print(f"Total Accuracy: {accuracy:.3f}%")

