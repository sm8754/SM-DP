
import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
weights_dict = torch.load("path_to_fine-tuned_plip.pth", map_location='cuda')
model.load_state_dict(weights_dict, strict=True)
model.eval()
image = preprocess(Image.open("/test.png")).unsqueeze(0).to(device)
text = clip.tokenize(["Tumor image of the liver.", "Tissue image of the liver.",
                      "Background image of the liver.", "ROI image of the liver."]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572 0]]