import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class MAX_model(nn.Module):
    def __init__(self, weights=None):
        super(MAX_model, self).__init__()
        
        # Backbone Değişimi: ViT-B/16
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        
        # Özellik Çıkarımı: Sınıflandırma katmanını Identity yaparak 768 boyutlu cls token vektörünü elde ediyoruz
        self.vit.heads = nn.Identity()
        
        if weights is not None:
            self.vit.load_state_dict(torch.load(weights))
            
        # Ağırlıkları Dondurma
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Lineer Katmanların (FC) Boyut Güncellemesi
        # 768 (ViT) + 4 (kutu koordinatları) = 772
        self.fc1 = nn.Linear(772, 512)
        self.fc2 = nn.Linear(512, 256)

    def forward_once(self, view_data):
        x, boxes, _ = view_data
        
        # Girdi x tensörünü doğrudan ViT üzerinden geçir
        x = self.vit(x)
        
        # Çıkan sonucu kutu koordinatları ile birleştir
        x = torch.cat((x, boxes[:, :4]), axis=1)
        
        # Ardından relu ve fc katmanlarından geçir
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Normalize et
        return F.normalize(x, p=2, dim=1)

    def forward(self, view_0, view_1):
        # import pdb; pdb.set_trace()
        num_view0_props = view_0[0].shape[0]
        
        # Her görünüm için ayrı embedding hesapla
        embedd_0 = self.forward_once(view_0) # MLO embedding: [N, 256]
        embedd_1 = self.forward_once(view_1) # CC embedding:  [M, 256]

        assert embedd_0.size() == embedd_1.size()
        
        # KRİTİK SATIR: Kosinüs benzerlik matrisi hesapla
        context = torch.matmul(embedd_0, embedd_1.transpose(-1, -2))
        # context boyutu: [N, M] — her MLO proposal'ı ile her CC proposal'ı arası benzerlik

        # En yüksek benzerliği seç + karşı görünümün güvenilirlik skoru ile çarp
        preds, _ = torch.max(context * view_1[1][:, 4].unsqueeze(1), axis=1)

        # Son skor = context-tabanlı skor + kendi orijinal skoru 
        preds = preds + view_0[1][:, 4]

        return preds

# torch.matmul — İki görünüm arasındaki çapraz dikkat (cross-attention) mekanizması
# Her proposal'ın skoru, karşı görünümdeki en benzer proposal ile güçlendiriliyor
# Bu "bağlam zenginleştirme" (context enhancement) sayesinde false positive'ler azalıyor