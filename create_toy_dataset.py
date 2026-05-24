"""
Toy Dataset Oluşturucu - CEN Demo İçin
=======================================
Bu script, CEN modelinin beklediği veri formatında sentetik mamografi 
verileri oluşturur. Her hasta klasöründe:
- MLO.png : Mediolateral Oblique görüntü (sentetik)
- CC.png  : Craniocaudal görüntü (sentetik)
- MLO_preds.txt : YOLO formatında tahmin kutuları (önceki detektörden)
- CC_preds.txt  : YOLO formatında tahmin kutuları
- MLO.txt : Ground truth (zemin gerçeği) - sadece malignant hastalar için
- CC.txt  : Ground truth - sadece malignant hastalar için

Veri formatı (YOLO): class x_center y_center width height
Pred formatı: x_center y_center width height confidence
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

random.seed(42)
np.random.seed(42)

def create_mammogram_like_image(width=1024, height=1024, view="MLO", has_lesion=False):
    """
    Mamografi benzeri sentetik bir görüntü oluşturur.
    - Arka plan koyu gri/siyah
    - Meme dokusu açık gri tonlarında  
    - Lezyon varsa parlak bir bölge eklenir
    """
    # Siyah arka plan
    img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(img)
    
    # Meme dokusunu simüle eden eliptik bölge
    if view == "MLO":
        # MLO görünüm - eğimli elips
        cx, cy = width // 3, height // 2
        rx, ry = width // 3, height // 2 - 50
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=80)
        # İç doku
        draw.ellipse([cx - rx + 40, cy - ry + 60, cx + rx - 60, cy + ry - 60], fill=100)
        draw.ellipse([cx - rx + 80, cy - ry + 120, cx + rx - 100, cy + ry - 120], fill=120)
    else:
        # CC görünüm - daha yuvarlak
        cx, cy = width // 2, height // 2
        rx, ry = width // 3, height // 3
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=80)
        draw.ellipse([cx - rx + 30, cy - ry + 40, cx + rx - 40, cy + ry - 30], fill=100)
        draw.ellipse([cx - rx + 70, cy - ry + 90, cx + rx - 80, cy + ry - 80], fill=120)
    
    # Rastgele doku paternleri ekle
    for _ in range(20):
        rx2 = random.randint(10, 40)
        ry2 = random.randint(10, 40)
        x = random.randint(cx - rx + 50, cx + rx - 50) if view == "CC" else random.randint(cx - rx + 50, cx)
        y = random.randint(cy - ry + 50, cy + ry - 50)
        brightness = random.randint(90, 140)
        draw.ellipse([x - rx2, y - ry2, x + rx2, y + ry2], fill=brightness)
    
    # Lezyon ekle (malignant hastalar için)
    lesion_box = None
    if has_lesion:
        lx = random.randint(cx - rx // 2, cx + rx // 4)
        ly = random.randint(cy - ry // 3, cy + ry // 4)
        lr = random.randint(20, 45)
        # Parlak, düzensiz şekilli lezyon
        draw.ellipse([lx - lr, ly - lr, lx + lr, ly + lr], fill=200)
        draw.ellipse([lx - lr + 5, ly - lr + 3, lx + lr - 3, ly + lr - 5], fill=220)
        # YOLO formatı: x_center, y_center, width, height (normalize)
        lesion_box = [lx / width, ly / height, (lr * 2.5) / width, (lr * 2.5) / height]
    
    # Gaussian blur ile yumuşatma
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    
    # RGB'ye çevir
    img_rgb = Image.merge('RGB', [img, img, img])
    
    return img_rgb, lesion_box


def create_pred_boxes(lesion_box, num_preds=5, has_lesion=False):
    """
    Önceki detektörün (ör. FocalNet/YOLO) ürettiği tahmin kutularını simüle eder.
    YOLO format: x_center y_center width height confidence
    """
    preds = []
    
    if has_lesion and lesion_box is not None:
        # True positive tahmin - lezyona yakın
        tp_box = [
            lesion_box[0] + np.random.uniform(-0.02, 0.02),
            lesion_box[1] + np.random.uniform(-0.02, 0.02),
            lesion_box[2] * np.random.uniform(0.8, 1.2),
            lesion_box[3] * np.random.uniform(0.8, 1.2),
            np.random.uniform(0.6, 0.95)  # Yüksek confidence
        ]
        preds.append(tp_box)
    
    # False positive tahminler
    for _ in range(num_preds - (1 if has_lesion else 0)):
        fp_box = [
            np.random.uniform(0.15, 0.65),
            np.random.uniform(0.2, 0.8),
            np.random.uniform(0.03, 0.08),
            np.random.uniform(0.03, 0.08),
            np.random.uniform(0.05, 0.4)  # Düşük confidence
        ]
        preds.append(fp_box)
    
    return np.array(preds)


def create_patient_folder(base_path, patient_id, is_malignant=True):
    """Bir hasta klasörü oluşturur."""
    patient_path = os.path.join(base_path, str(patient_id))
    os.makedirs(patient_path, exist_ok=True)
    
    for view in ["MLO", "CC"]:
        # Görüntü oluştur
        img, lesion_box = create_mammogram_like_image(
            width=512, height=512, 
            view=view, 
            has_lesion=is_malignant
        )
        img_path = os.path.join(patient_path, f"{view}.png")
        img.save(img_path)
        print(f"  [OK] {view}.png olusturuldu ({img.size[0]}x{img.size[1]})")
        
        # Tahmin kutuları
        pred_boxes = create_pred_boxes(lesion_box, num_preds=4, has_lesion=is_malignant)
        pred_path = os.path.join(patient_path, f"{view}_preds.txt")
        np.savetxt(pred_path, pred_boxes, fmt='%.6f')
        print(f"  [OK] {view}_preds.txt olusturuldu ({len(pred_boxes)} tahmin)")
        
        # Ground truth (sadece malignant hastalar)
        if is_malignant and lesion_box is not None:
            gt_data = np.array([[0] + lesion_box])  # class + x,y,w,h
            gt_path = os.path.join(patient_path, f"{view}.txt")
            np.savetxt(gt_path, gt_data, fmt='%.6f')
            print(f"  [OK] {view}.txt (ground truth) olusturuldu")
    
    # Dosya sayısı kontrolü
    file_count = len(os.listdir(patient_path))
    status = "MALIGNANT" if is_malignant else "BENIGN"
    print(f"  [DIR] Hasta {patient_id}: {status} ({file_count} dosya)")
    if is_malignant:
        assert file_count == 6, f"Malignant hasta {file_count} dosya olmalı 6!"
    else:
        assert file_count == 4, f"Benign hasta {file_count} dosya olmalı 4!"


def main():
    # === TRAIN VERİSETİ ===
    train_path = os.path.join("DEMO_DATA", "TRAIN")
    os.makedirs(train_path, exist_ok=True)
    
    print("=" * 60)
    print("CEN Demo - Toy Dataset Olusturucu")
    print("=" * 60)
    
    print("\nTRAIN Veriseti Olusturuluyor...")
    print("-" * 40)
    
    # Malignant hastalar (6 dosya: MLO.png, CC.png, MLO_preds.txt, CC_preds.txt, MLO.txt, CC.txt)
    for pid in [101, 102, 103]:
        print(f"\n[+] Hasta {pid} (Malignant):")
        create_patient_folder(train_path, pid, is_malignant=True)
    
    # Benign hastalar (4 dosya: MLO.png, CC.png, MLO_preds.txt, CC_preds.txt)
    for pid in [201, 202, 203]:
        print(f"\n[-] Hasta {pid} (Benign):")
        create_patient_folder(train_path, pid, is_malignant=False)
    
    # === TEST VERISETI ===
    test_path = os.path.join("DEMO_DATA", "TEST")
    os.makedirs(test_path, exist_ok=True)
    
    print(f"\n\nTEST Veriseti Olusturuluyor...")
    print("-" * 40)
    
    for pid in [301, 302]:
        print(f"\n[+] Hasta {pid} (Malignant):")
        create_patient_folder(test_path, pid, is_malignant=True)
    
    for pid in [401]:
        print(f"\n[-] Hasta {pid} (Benign):")
        create_patient_folder(test_path, pid, is_malignant=False)
    
    print("\n" + "=" * 60)
    print("Toy Dataset basariyla olusturuldu!")
    print(f"   TRAIN: {train_path} ({len(os.listdir(train_path))} hasta)")
    print(f"   TEST:  {test_path} ({len(os.listdir(test_path))} hasta)")
    print("=" * 60)
    
    # Veri yapisini goster
    print("\nBeklenen Dizin Yapisi:")
    print("DEMO_DATA/")
    for split in ["TRAIN", "TEST"]:
        split_path = os.path.join("DEMO_DATA", split)
        print(f"├── {split}/")
        folders = sorted(os.listdir(split_path))
        for i, f in enumerate(folders):
            prefix = "│   └──" if i == len(folders) - 1 else "│   ├──"
            fp = os.path.join(split_path, f)
            files = os.listdir(fp)
            print(f"{prefix} {f}/ ({len(files)} dosya: {', '.join(sorted(files))})")


if __name__ == '__main__':
    main()
