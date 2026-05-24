"""
CEN Demo Test ve Gorsellestirme Scripti
========================================
Egitilen modeli test dataseti uzerinde calistirip 
gorsel ciktilar uretir. Cikti gorselleri:
- Yesil kutular: Modelin dogrulanmis tahminleri (True Positives)
- Kirmizi kutular: Elenen hatali tahminler (False Positives)  
- Mavi kutular: Doktorun isaretledigi gercek kanser alanlari (Ground Truth)
"""

import numpy as np
import os
import sys
from data import PAIRDataset
from models import MAX_model
from torch import nn
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from tqdm import tqdm
import random
import cv2


# ============================================
# KONFIGÜRASYON
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "DEMO_DATA", "TEST")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "demo_output", "test_results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read_data(folder_path):
    """Bir hasta klasorunun tahmin ve ground truth verilerini okur."""
    pred_list = []
    images = glob.glob(folder_path + "/*.png")
    for j, image_path in enumerate(images):
        item_info = {}
        image_view = "MLO" if ("MLO" in image_path) else "CC"
        preds_path = image_path[:-4] + "_preds.txt"
        preds = torch.tensor(np.loadtxt(preds_path))
        if preds.shape[0] == 0:
            preds = torch.tensor(np.array([[0, 0, 0, 0, -1]]).astype(np.float32))
        if len(preds.shape) == 1:
            preds = preds.unsqueeze(0)
        output = {
            "boxes": preds[:, :4],
            "scores": preds[:, 4],
            "labels": torch.zeros(preds.shape[0])
        }
        target_path = image_path[:-4] + ".txt"
        if os.path.isfile(target_path):
            targets = torch.tensor(np.loadtxt(target_path))
            if targets.shape[0] != 0:
                if len(targets.shape) == 1:
                    targets = targets.unsqueeze(0)
                targets = targets[:, 1:]
            else:
                targets = torch.tensor([])
        else:
            targets = torch.tensor([])
        item_info['pred'] = output
        item_info['target'] = {"boxes": targets}
        item_info["view"] = image_view
        item_info["img_path"] = image_path
        pred_list.append(item_info)
    return pred_list


def change_confs(folder_path, mlo_scores, cc_scores):
    """Model skorlarini guncelle."""
    num_props = mlo_scores.shape[0]
    folder_data = read_data(folder_path[0])
    if folder_data[0]["view"] == "CC":
        folder_data.append(folder_data.pop(0))

    folder_data[0]["pred"]["scores"] = folder_data[0]["pred"]["scores"][:num_props]
    folder_data[1]["pred"]["scores"] = folder_data[1]["pred"]["scores"][:num_props]
    folder_data[0]["pred"]["boxes"] = folder_data[0]["pred"]["boxes"][:num_props]
    folder_data[1]["pred"]["boxes"] = folder_data[1]["pred"]["boxes"][:num_props]
    folder_data[0]["pred"]["labels"] = folder_data[0]["pred"]["labels"][:num_props]
    folder_data[1]["pred"]["labels"] = folder_data[1]["pred"]["labels"][:num_props]

    folder_data[0]["pred"]["scores"] = mlo_scores
    folder_data[1]["pred"]["scores"] = cc_scores

    return folder_data


def save_visualization(img_path, output_folder, pred_boxes, gt_boxes, pred_scores, threshold=0.1):
    """
    Goruntunun uzerine tahmin ve ground truth kutularini cizer.
    - Yesil: Modelin kabul ettigi tahminler (score > threshold)
    - Kirmizi: Modelin eledigi tahminler (score <= threshold)
    - Mavi: Ground truth (doktorun isaretledigi gercek bolge)
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"  UYARI: Goruntu okunamadi: {img_path}")
        return
    
    h, w = img.shape[:2]
    img_name = os.path.basename(img_path)
    
    # Tum tahmin kutularini ciz
    for i, box in enumerate(pred_boxes):
        x = int((box[0] - box[2] / 2) * w)
        y = int((box[1] - box[3] / 2) * h)
        bw = int(box[2] * w)
        bh = int(box[3] * h)
        
        score = float(pred_scores[i]) if i < len(pred_scores) else 0
        
        if score > threshold:
            # YESIL: Kabul edilen tahmin (True Positive aday)
            color = (0, 255, 0)
            thickness = 3
            label = f"Pred: {score:.3f}"
        else:
            # KIRMIZI: Elenen tahmin (False Positive)
            color = (0, 0, 255)
            thickness = 2
            label = f"FP: {score:.3f}"
        
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, thickness)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Ground truth kutularini ciz (MAVI)
    for gt_box in gt_boxes:
        x = int((gt_box[0] - gt_box[2] / 2) * w)
        y = int((gt_box[1] - gt_box[3] / 2) * h)
        bw = int(gt_box[2] * w)
        bh = int(gt_box[3] * h)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (255, 0, 0), 3)
        cv2.putText(img, "GT (Gercek)", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, img_name)
    cv2.imwrite(output_path, img)
    return output_path


def demo_test(model_path=None):
    """Demo test fonksiyonu."""
    print("\n" + "=" * 60)
    print("CEN - DEMO TEST VE GORSELLESTIRME")
    print("=" * 60)
    print(f"[CIHAZ] {DEVICE}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    sigmoid = nn.Sigmoid()
    
    # Veri yukle
    print(f"\n[1/3] Test verisi yukleniyor: {TEST_DATA_PATH}")
    dataset = PAIRDataset(pairs_path=TEST_DATA_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    print(f"  Test ornekleri: {len(dataset)}")
    
    # Model yukle
    print(f"\n[2/3] Model yukleniyor...")
    model = MAX_model(weights=None).to(DEVICE)
    
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"  Yuklenen agirlik: {model_path}")
    else:
        # Model ağırlığı yoksa, en son demo_output'taki modeli bul
        demo_models = glob.glob(os.path.join(SCRIPT_DIR, "demo_output", "best_model_*.pth"))
        if demo_models:
            latest = sorted(demo_models)[-1]
            model.load_state_dict(torch.load(latest, map_location=DEVICE))
            print(f"  Yuklenen agirlik: {latest}")
        else:
            print("  UYARI: Model agirligi bulunamadi! Pretrained ResNet50 ile devam ediliyor.")
    
    model.eval()
    
    # Test
    print(f"\n[3/3] Test ve gorsellestirme basliyor...")
    print("-" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pred_list = []
    with torch.no_grad():
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader, desc="Test")):
            mlo_data[0] = mlo_data[0].squeeze(0).to(DEVICE)
            mlo_data[1] = mlo_data[1].squeeze(0).to(DEVICE)
            cc_data[0] = cc_data[0].squeeze(0).to(DEVICE)
            cc_data[1] = cc_data[1].squeeze(0).to(DEVICE)

            max_props = min(min(mlo_data[0].shape[0], cc_data[0].shape[0]), 25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            preds_mlo = model(mlo_data, cc_data)
            preds_mlo = sigmoid(preds_mlo)

            preds_cc = model(cc_data, mlo_data)
            preds_cc = sigmoid(preds_cc)

            folder_data = change_confs(mlo_data[2], preds_mlo.cpu(), preds_cc.cpu())
            pred_list += folder_data
    
    # Gorsellestirme
    print(f"\n  Gorseller olusturuluyor...")
    threshold = 0.3
    saved_count = 0
    
    for item in pred_list:
        img_path = item["img_path"]
        patient_id = os.path.basename(os.path.dirname(img_path))
        
        select_mask = item["pred"]["scores"] > threshold
        accepted_boxes = item["pred"]["boxes"][select_mask]
        rejected_boxes = item["pred"]["boxes"][~select_mask]
        gt_boxes = item["target"]["boxes"]
        
        # Tum kutulari birlikte goster
        all_boxes = item["pred"]["boxes"]
        all_scores = item["pred"]["scores"]
        
        patient_dir = os.path.join(OUTPUT_DIR, patient_id)
        saved_path = save_visualization(
            img_path, patient_dir, 
            all_boxes.numpy(), 
            gt_boxes.numpy() if len(gt_boxes) > 0 else [],
            all_scores.numpy(),
            threshold=threshold
        )
        if saved_path:
            saved_count += 1
    
    print(f"  {saved_count} gorsel kaydedildi: {OUTPUT_DIR}/")
    
    # Ozet istatistikler
    print("\n" + "=" * 60)
    print("TEST SONUCLARI OZET")
    print("=" * 60)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for item in pred_list:
        gt_boxes = item["target"]["boxes"]
        scores = item["pred"]["scores"]
        pred_boxes = item["pred"]["boxes"][scores > threshold]
        
        if len(gt_boxes) > 0 and len(pred_boxes) > 0:
            total_tp += 1
        elif len(gt_boxes) == 0 and len(pred_boxes) > 0:
            total_fp += 1
        elif len(gt_boxes) > 0 and len(pred_boxes) == 0:
            total_fn += 1
    
    print(f"  Threshold: {threshold}")
    print(f"  True Positive:  {total_tp} (Dogrulanan tahminler)")
    print(f"  False Positive: {total_fp} (Hatali tahminler)")
    print(f"  False Negative: {total_fn} (Kacirilan lezyonlar)")
    print(f"\n  Gorsel ciktilar: {OUTPUT_DIR}/")
    print("  Yesil kutu = Modelin dogruladigi tahmin")
    print("  Kirmizi kutu = Modelin eledigi hatali tahmin")
    print("  Mavi kutu = Ground Truth (Doktorun isaretledigi bolge)")
    print("=" * 60)


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    demo_test(model_path)
