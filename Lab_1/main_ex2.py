import cv2
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import Imagenette
from torch.utils.data import DataLoader
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Class Activation Maps for pretrained ResNet-18 on Imagenette dataset"
    )
    parser.add_argument("--class_index", type=int, default=0,
                        help="Class index (0-9) for Imagenette")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to analyze per class")
    parser.add_argument("--analyze_all", action='store_true',
                        help="Analyze samples from all classes")
    parser.add_argument("--save_results", action='store_true',
                        help="Save CAM visualizations to files")
    return parser.parse_args()

IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute"
]

IMAGENETTE_TO_IMAGENET = {
    0: 0, 1: 217, 2: 482, 3: 491, 4: 497,
    5: 566, 6: 569, 7: 571, 8: 574, 9: 701
}

class CAMVisualizer:
    def __init__(self, device):
        self.device = device
        self.net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.net.eval()
        self.features_blobs = []
        self.finalconv_name = "layer4"
        self.net._modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)
        self.weight_softmax = self.net.fc.weight.data.cpu().numpy()
        self.imagenet_classes = ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
        self.preprocess = transforms.Compose([
            transforms.Resize(160),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def hook_feature(self, module, input, output):
        self.features_blobs.append(output.data.cpu().numpy())

    def generate_cam(self, feature_conv, class_idx, img_size=(224,224)):
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = self.weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            if np.max(cam) > 0:
                cam_img = cam / np.max(cam)
            else:
                cam_img = cam
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, img_size))
        return output_cam

    def predict_and_visualize(self, img_pil, true_label=None, save_path=None):
        self.features_blobs.clear()
        img_tensor = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        logit = self.net(img_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        probs = probs.cpu().numpy()
        idx = idx.cpu().numpy()

        print("\nTop-5 predictions:")
        for i in range(5):
            print(f"{probs[i]:.3f} -> {self.imagenet_classes[idx[i]]}")

        CAMs = self.generate_cam(self.features_blobs[0], [idx[0]])
        self.visualize_cam(img_pil, CAMs[0], idx[0], probs[0], true_label, save_path)
        return idx[0], probs[0]

    def visualize_cam(self, img_pil, cam, pred_class_idx, confidence, true_label=None, save_path=None):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_cv = cv2.resize(img_cv, (224,224))
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        result = heatmap * 0.4 + img_cv * 0.6
        result_display = cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1,3, figsize=(15,5))
        axes[0].imshow(img_pil)
        title = "Original Image"
        if true_label is not None:
            title += f"\nTrue: {IMAGENETTE_CLASSES[true_label]}"
        axes[0].set_title(title)
        axes[0].axis('off')

        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title(f"CAM\nPredicted: {IMAGENETTE_CLASSES[pred_class_idx]}\nConfidence: {confidence:.3f}")
        axes[1].axis('off')

        axes[2].imshow(result_display)
        axes[2].set_title("CAM Overlay")
        axes[2].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved CAM visualization to {save_path}")
        plt.show()
        return result_display

def load_imagenette_data():
    transform = transforms.Compose([
        transforms.Resize(160),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    val_dataset = Imagenette(root='./data', split='val', download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    return val_loader, val_dataset

def denormalize_tensor(tensor):
    tensor = torch.clamp(tensor,0,1)
    transform = transforms.ToPILImage()
    return transform(tensor)

def analyze_class_samples(cam_visualizer, val_loader, class_idx, num_samples=5, save_results=False):
    print(f"\n{'='*60}")
    print(f"Analyzing Class: {IMAGENETTE_CLASSES[class_idx]}")
    print(f"{'='*60}")
    samples_found = 0
    for batch_idx, (data, target) in enumerate(val_loader):
        if target.item() == class_idx and samples_found < num_samples:
            img_pil = denormalize_tensor(data.squeeze(0))
            save_path = None
            if save_results:
                os.makedirs('cam_results', exist_ok=True)
                save_path = f'cam_results/cam_{IMAGENETTE_CLASSES[class_idx]}_sample_{samples_found+1}.png'
            cam_visualizer.predict_and_visualize(img_pil, true_label=target.item(), save_path=save_path)
            samples_found += 1

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cam_visualizer = CAMVisualizer(device)
    val_loader, val_dataset = load_imagenette_data()
    print(f"Dataset loaded: {len(val_dataset)} validation samples")

    if args.analyze_all:
        for class_idx in range(len(IMAGENETTE_CLASSES)):
            analyze_class_samples(cam_visualizer, val_loader, class_idx, args.num_samples, args.save_results)
    else:
        if args.class_index < 0 or args.class_index >= len(IMAGENETTE_CLASSES):
            print(f"Error: class_index must be between 0 and {len(IMAGENETTE_CLASSES)-1}")
            return
        analyze_class_samples(cam_visualizer, val_loader, args.class_index, args.num_samples, args.save_results)

if __name__ == "__main__":
    main()
