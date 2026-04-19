import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import cv2
import albumentations as A
from tqdm import tqdm
from utilities.UASS_net_factory import net_factory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset_name = 'dataset name'
IMAGE_PATH_test = f'/root/data/{dataset_name}/test_images/'
MASK_PATH_test = f'/root/data/{dataset_name}/test_annot/'

models_to_evaluate = [
    {
        "name": "",
        "path": ".pth"
    },
]
def pixel_accuracy(pred_mask, true_mask):
    correct = torch.eq(pred_mask, true_mask).int()
    return float(correct.sum()) / float(correct.numel())

def mIoU(pred_logits, true_mask, n_classes=2, smooth=1e-10):
    pred_mask = torch.argmax(F.softmax(pred_logits, dim=0), dim=0)
    pred_mask = pred_mask.contiguous().view(-1)
    true_mask = true_mask.contiguous().view(-1)
    iou_per_class = []
    for clas in range(1, n_classes):
        true_class = pred_mask == clas
        true_label = true_mask == clas
        if true_label.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            union = torch.logical_or(true_class, true_label).sum().float().item()
            iou = (intersect + smooth) / (union + smooth)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class) if iou_per_class else 0.0

def mDice(pred_logits, true_mask, n_classes=2, smooth=1e-10):
    pred_mask = torch.argmax(F.softmax(pred_logits, dim=0), dim=0)
    pred_mask = pred_mask.contiguous().view(-1)
    true_mask = true_mask.contiguous().view(-1)
    dice_per_class = []
    for clas in range(1, n_classes):
        true_class = pred_mask == clas
        true_label = true_mask == clas
        if true_label.long().sum().item() == 0:
            dice_per_class.append(np.nan)
        else:
            intersect = torch.logical_and(true_class, true_label).sum().float().item()
            total_sum = true_class.sum().float().item() + true_label.sum().float().item()
            dice = (2 * intersect + smooth) / (total_sum + smooth)
            dice_per_class.append(dice)
    return np.nanmean(dice_per_class) if dice_per_class else 0.0

def precision_recall(pred_mask, true_mask, n_classes=2):
    precisions, recalls, ff = [], [], []
    for class_id in range(n_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        tp = (pred_class & true_class).sum().float().item()
        fp = (pred_class & ~true_class).sum().float().item()
        fn = (~pred_class & true_class).sum().float().item()
        precision = (tp + 1e-10) / (tp + fp + 1e-10)
        recall = (tp + 1e-10) / (tp + fn + 1e-10)
        precisions.append(precision)
        recalls.append(recall)
    return np.mean(precisions), np.mean(recalls), np.mean(ff)

def mPA(pred_mask, true_mask, n_classes=2):
    class_accuracies = []
    for class_id in range(n_classes):
        true_positive = ((pred_mask == class_id) & (true_mask == class_id)).sum().float()
        total_positive = (true_mask == class_id).sum().float()
        if total_positive == 0:
            if pred_mask.sum() == 0:
                class_accuracies.append(1.0)
            else:
                class_accuracies.append(0.0)
        else:
            class_accuracies.append((true_positive / total_positive).item())
    return np.mean(class_accuracies)

def fwIoU(pred_mask, true_mask, n_classes=2):
    fw_iou = 0.0
    total_pixels = true_mask.numel()
    for class_id in range(n_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        intersection = (pred_class & true_class).sum().float()
        union = (pred_class | true_class).sum().float()
        iou = (intersection + 1e-10) / (union + 1e-10)
        class_frequency = true_class.sum().item() / total_pixels
        fw_iou += iou * class_frequency
    return fw_iou.item()

class TestDataset(Dataset):
    def __init__(self, img_path, mask_path, img_ids, transform=None, test_transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.X = img_ids
        self.transform = transform
        self.test_transform = test_transform
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img_name = self.X[idx]
        img = cv2.imread(os.path.join(self.img_path, img_name + '.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_path, img_name + '.png'), cv2.IMREAD_GRAYSCALE)
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        if self.test_transform:
            img = self.test_transform(img)  
        mask = torch.from_numpy(mask).long()
        return img, mask

def evaluate_model(model, test_loader):
    model.eval()
    all_metrics = {
        "acc": [], "miou": [], "mdice": [], "precision": [],
        "recall": [], "mpa": [], "fwiou": []
    }
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs, _, _, _ = model(images)
            for i in range(images.size(0)):
                output_logits = outputs[i]
                true_mask = masks[i]
                pred_mask = torch.argmax(output_logits, dim=0)
                all_metrics["acc"].append(pixel_accuracy(pred_mask, true_mask))
                all_metrics["miou"].append(mIoU(output_logits, true_mask))
                all_metrics["mdice"].append(mDice(output_logits, true_mask))
                p, r, _ = precision_recall(pred_mask, true_mask)
                all_metrics["precision"].append(p)
                all_metrics["recall"].append(r)
                all_metrics["mpa"].append(mPA(pred_mask, true_mask))
                all_metrics["fwiou"].append(fwIoU(pred_mask, true_mask))
    mean_metrics = {k: np.nanmean(v) * 100 for k, v in all_metrics.items()}
    return mean_metrics

if __name__ == '__main__':
    test_image_ids = [f.split('.')[0] for f in os.listdir(IMAGE_PATH_test)]
    
    t_resize = A.Resize(256, 256, interpolation=cv2.INTER_NEAREST)
    t_test = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    test_set = TestDataset(IMAGE_PATH_test, MASK_PATH_test, test_image_ids, transform=t_resize, test_transform=t_test)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=4)
    results = []

    for model_info in models_to_evaluate:
        print(f"\n--- Testing model: {model_info['name']} ---")
        model = net_factory(net_type='unet_uaps', in_chns=3, class_num=2)
        model = nn.DataParallel(model)
        try:
            checkpoint = torch.load(model_info['path'], map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
            model.to(device)
        except Exception as e:
            print(f"Error loading model {model_info['name']}: {e}")
            continue

        metrics = evaluate_model(model, test_loader)
        metrics['Model'] = model_info['name']
        results.append(metrics)

    header = "Model\tAccuracy(%)\tmDice(%)\tmIoU(%)\tPrecision(%)\tRecall(%)\tmPA(%)\tFWIoU(%)"
    print(header)
    
    for res in results:
        row = (
            f"{res['Model']}\t"
            f"{res['acc']:.4f}\t"
            f"{res['mdice']:.4f}\t"
            f"{res['miou']:.4f}\t"
            f"{res['precision']:.4f}\t"
            f"{res['recall']:.4f}\t"
            f"{res['mpa']:.4f}\t"
            f"{res['fwiou']:.4f}"
        )
        print(row)
