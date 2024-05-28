import torch
import clip
from collections import Counter
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
class Predictor:
    def __init__(self, model_name="ViT-B/32"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.device = device
        self.text = ["Tumor image of the liver.",
                     "Tissue image of the liver.",
                     "Background image of the liver.",
                     "ROI image of the liver."]
        self.classes = ['Tumor', 'Tissue', 'Background', 'ROI']

        self.text_inputs = torch.cat([clip.tokenize(f"{c} image of the liver.") for c in self.classes]).to(device)

        self.text_features = self.model.encode_text(self.text_inputs)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    def predict(self, wj, image_path, save_flag = False):
        if wj == 10:
            self.block = 128
            weights_dict = torch.load(
                "path_to_trained_clip_for_hcc.pth",
                map_location='cuda')
            self.model.load_state_dict(weights_dict, strict=True)
        else:
            self.block = 256
            weights_dict = torch.load(
                "path_to_trained_clip_for_roi.pth",
                map_location='cuda')
            self.model.load_state_dict(weights_dict, strict=True)
        original_image = Image.open(image_path)
        width, height = original_image.size
        draw = ImageDraw.Draw(original_image)
        rows, cols = height // self.block, width // self.block
        predictions = []
        mask_image = Image.new('L', (width, height), 0)
        for i in range(rows):
            for j in range(cols):
                left, upper, right, lower = j * self.block, i * self.block, (j + 1) * self.block, (i + 1) * self.block
                image_block = original_image.crop((left, upper, right, lower))
                image = self.preprocess(image_block).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    image_features = self.model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
                    max_index = similarity[0].argmax().item()

                predicted_class = self.classes[max_index]
                predictions.append(predicted_class)

                if predicted_class == 'Tumor':
                    block_color = Image.new('L', (self.block, self.block), 255)
                elif predicted_class == 'ROI':
                    block_color = Image.new('L', (self.block, self.block), 128)
                else:
                    block_color = Image.new('L', (self.block, self.block), 0)

                mask_image.paste(block_color, (left, upper))

        mask_image_path = os.path.splitext(image_path)[0] + '_mask.png'
        mask_image.save(mask_image_path)

        most_common_prediction = Counter(predictions).most_common(1)[0][0]
        print(f"{most_common_prediction} image of the liver.\n")

        return most_common_prediction, mask_image_path
