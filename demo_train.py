"""
CEN Demo Egitim Scripti
========================
Bu script, sinifta canli demo icin tasarlanmistir.
Orijinal train.py'nin DEMO_DATA/ toy dataseti ile calisacak 
sekilde uyarlanmis halidir.

Degisiklikler:
- Epoch sayisi 2 (hizli demo)
- num_workers=0 (Windows uyumlu)
- CPU/CUDA otomatik secim
- DEMO_DATA/ dizinini kullanir
- ViT-B/16 pretrained agirliklar dogrudan torchvision'dan alinir
"""

import numpy as np
import os
from data import PAIRDataset
from models import MAX_model
from torch import nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw
import glob
from tqdm import tqdm

import random
import matplotlib
matplotlib.use('Agg')  # GUI gerektirmeyen backend
import matplotlib.pyplot as plt

# ============================================
# KONFIGÜRASYON - DEMO ICIN AYARLAR
# ============================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_EPOCHS = 2          # Sinifta hizli demo icin 2 epoch yeterli
LEARNING_RATE = 0.000001  # Orijinal makaledeki ogrenme orani
TRAIN_DATA_PATH = os.path.join(SCRIPT_DIR, "DEMO_DATA", "TRAIN")
TEST_DATA_PATH = os.path.join(SCRIPT_DIR, "DEMO_DATA", "TEST")
EXP_NAME = os.path.join(SCRIPT_DIR, "demo_output")

# Cihaz secimi
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n[CIHAZ] Kullanilan cihaz: {DEVICE}")
if DEVICE == "cuda":
    print(f"[CIHAZ] GPU: {torch.cuda.get_device_name(0)}")


def val(dataloader, model):
    """Validation loss hesaplama."""
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()

    with torch.no_grad():
        loss_item = 0
        for i, (mlo_data, cc_data) in enumerate(tqdm(dataloader, desc="Val ")):
            mlo_data[0] = mlo_data[0].squeeze(0).to(DEVICE)
            mlo_data[1] = mlo_data[1].squeeze(0).to(DEVICE)
            cc_data[0] = cc_data[0].squeeze(0).to(DEVICE)
            cc_data[1] = cc_data[1].squeeze(0).to(DEVICE)

            max_props = min(min(mlo_data[0].shape[0], cc_data[0].shape[0]), 25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:, 5]
            preds = sigmoid(preds)
            loss = loss_fn(preds, targets)

            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:, 5]
            preds = sigmoid(preds)
            loss = loss + loss_fn(preds, targets)

            loss_item += (loss / (mlo_data[0].shape[0] * cc_data[0].shape[0])).item()
        return loss_item


def train_demo():
    """Ana demo egitim fonksiyonu."""
    print("\n" + "=" * 60)
    print("CEN - Context Enhanced Network - DEMO EGITIMI")
    print("=" * 60)
    
    # -------------------------------------------
    # 1. VERi YUKLE
    # -------------------------------------------
    print(f"\n[1/4] Veri yukleniyor...")
    print(f"  Train: {TRAIN_DATA_PATH}")
    print(f"  Test:  {TEST_DATA_PATH}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_dataset = PAIRDataset(pairs_path=TRAIN_DATA_PATH, transform=transform)
    val_dataset = PAIRDataset(pairs_path=TEST_DATA_PATH, transform=transform)
    
    # Windows'ta num_workers=0 olmali (fork destegi yok)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0)
    
    print(f"  Train ornekleri: {len(train_dataset)}")
    print(f"  Val ornekleri:   {len(val_dataset)}")
    
    # -------------------------------------------
    # 2. MODELI OLUSTUR 
    # -------------------------------------------
    print(f"\n[2/4] Model olusturuluyor...")
    print("  Backbone: ViT-B/16 (ImageNet pretrained)")
    print("  FC Katmanlari: 772 -> 512 -> 256")
    
    # ViT-B/16 pretrained agirliklarini dogrudan kullan
    # (model_weights klasorune gerek yok)
    model = MAX_model(weights=None).to(DEVICE)
    
    loss_fn = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Toplam parametre: {total_params:,}")
    print(f"  Egitilir parametre: {trainable_params:,}")
    
    # -------------------------------------------
    # 3. EGITIM DONGUSU
    # -------------------------------------------
    print(f"\n[3/4] Egitim basliyor... ({DEMO_EPOCHS} epoch)")
    print("-" * 60)
    
    os.makedirs(EXP_NAME, exist_ok=True)
    best_loss = 9999
    train_losses = []
    val_losses = []
    
    for epoch in range(DEMO_EPOCHS):
        loss_item = 0
        model.train()
        
        for j, (mlo_data, cc_data) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{DEMO_EPOCHS}")):
            mlo_data[0] = mlo_data[0].squeeze(0).to(DEVICE)
            mlo_data[1] = mlo_data[1].squeeze(0).to(DEVICE)
            cc_data[0] = cc_data[0].squeeze(0).to(DEVICE)
            cc_data[1] = cc_data[1].squeeze(0).to(DEVICE)

            max_props = min(min(mlo_data[0].shape[0], cc_data[0].shape[0]), 25)
            mlo_data[0] = mlo_data[0][:max_props]; mlo_data[1] = mlo_data[1][:max_props]
            cc_data[0] = cc_data[0][:max_props]; cc_data[1] = cc_data[1][:max_props]

            loss = 0
            optimizer.zero_grad()
            
            # ----- ILERI YAYILIM (Forward Pass) -----
            # MLO gorunumunu ana, CC'yi baglamsal (context) goruntoleme olarak kullan
            preds = model(mlo_data, cc_data)
            targets = mlo_data[1][:, 5]
            preds = sigmoid(preds)
            loss = loss + loss_fn(preds, targets)

            # CC gorunumunu ana, MLO'yu baglamsal olarak kullan
            preds = model(cc_data, mlo_data)
            targets = cc_data[1][:, 5]
            preds = sigmoid(preds)
            loss = loss + loss_fn(preds, targets)

            # Normalize et
            loss = loss / (mlo_data[0].shape[0] * cc_data[0].shape[0])
            
            # ----- GERI YAYILIM (Backward Pass) -----
            loss.backward()
            optimizer.step()
            loss_item += loss.item()
        
        # Validation
        model.eval()
        val_loss_item = val(val_loader, model)
        
        print(f"  >> Epoch: {epoch+1}/{DEMO_EPOCHS}  |  Train Loss: {loss_item:.4f}  |  Val Loss: {val_loss_item:.4f}")
        
        train_losses.append(loss_item)
        val_losses.append(val_loss_item)
        
        # En iyi modeli kaydet
        if val_loss_item < best_loss:
            best_loss = val_loss_item
            model_path = os.path.join(EXP_NAME, f"best_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"  >> En iyi model kaydedildi: {model_path}")
    
    # -------------------------------------------
    # 4. LOSS GRAFIGI
    # -------------------------------------------
    print(f"\n[4/4] Sonuclar kaydediliyor...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    epochs_list = list(range(1, DEMO_EPOCHS + 1))
    
    ax1.plot(epochs_list, train_losses, 'b-o', label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_list, val_losses, 'r-o', label='Validation Loss', linewidth=2)
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.subplots_adjust(hspace=0.4)
    plot_path = os.path.join(EXP_NAME, 'demo_loss_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Loss grafigi: {plot_path}")
    
    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI!")
    print(f"  Son Train Loss: {train_losses[-1]:.4f}")
    print(f"  Son Val Loss:   {val_losses[-1]:.4f}")
    print(f"  Ciktilar:       {EXP_NAME}/")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    model = train_demo()
