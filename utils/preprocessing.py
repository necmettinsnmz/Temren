"""
Görüntü ön işleme (preprocessing) fonksiyonları.

Yarışma esnasında gelen görüntüler farklı aydınlatma koşullarında,
farklı kameralardan (termal, RGB, uydu) ve farklı irtifalardan olabilir.
Bu modül, eşleştirme başarısını artırmak için görüntüleri normalize eder.
"""

import cv2
import numpy as np
from utils.logger import log


def enhance_image(image, apply_clahe=True, apply_denoise=False):
    """
    Görüntüyü eşleştirme için optimize eder.
    
    - CLAHE: Kontrast sınırlı histogram eşitleme. Gölgeli/karanlık bölgelerdeki
      detayları ortaya çıkarır. Feature extraction başarısını %15-25 artırır.
    - Denoise: Gürültü azaltma. Termal kameralarda etkili.
    """
    result = image.copy()
    
    if apply_clahe:
        result = apply_clahe_enhancement(result)
    
    if apply_denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 5, 5, 7, 21)
    
    return result


def apply_clahe_enhancement(image, clip_limit=2.0, tile_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Lokal kontrast iyileştirmesi yaparak farklı aydınlatma koşullarında
    feature noktalarının daha tutarlı çıkmasını sağlar.
    """
    if len(image.shape) == 2:
        # Gri tonlama
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(image)
    
    # Renkli görüntü -> LAB renk uzayına çevir, sadece L kanalına CLAHE uygula
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)


def resize_with_aspect_ratio(image, max_dim=1024):
    """
    En-boy oranını koruyarak görüntüyü yeniden boyutlandırır.
    Çok büyük görüntülerde işlem süresini kısaltır.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image, 1.0
    
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def compute_match_confidence(matches, kpts0, kpts1, homography_mask):
    """
    Eşleşmenin kalitesini 0-1 arasında bir güven skoru olarak hesaplar.
    
    Faktörler:
    - inlier_ratio: RANSAC sonrası doğru eşleşmelerin oranı
    - match_density: Eşleşen nokta sayısının toplam nokta sayısına oranı
    - spatial_spread: Noktaların obje üzerindeki dağılımı (uniform daha iyi)
    """
    if homography_mask is None or len(matches) == 0:
        return 0.0
    
    inlier_mask = homography_mask.ravel().astype(bool)
    num_inliers = np.sum(inlier_mask)
    
    # İnlier oranı (0-1)
    inlier_ratio = num_inliers / len(matches) if len(matches) > 0 else 0.0
    
    # Eşleşme yoğunluğu (normalize edilmiş)
    match_density = min(1.0, num_inliers / 50.0)  # 50+ eşleşme = tam puan
    
    # Mekansal yayılım (noktalar ne kadar dağılmışsa o kadar güvenilir)
    if num_inliers >= 4:
        inlier_pts = kpts1[inlier_mask]
        spread_x = np.std(inlier_pts[:, 0])
        spread_y = np.std(inlier_pts[:, 1])
        spatial_spread = min(1.0, (spread_x * spread_y) / 10000.0)
    else:
        spatial_spread = 0.0
    
    # Ağırlıklı toplam
    confidence = 0.5 * inlier_ratio + 0.3 * match_density + 0.2 * spatial_spread
    
    return float(np.clip(confidence, 0.0, 1.0))


def verify_homography(H, image_shape):
    """
    Bulunan homografi matrisinin geometrik olarak mantıklı olup olmadığını kontrol eder.
    Yanlış eşleşmelerden kaynaklanan saçma kutuları (ters dönen, çok küçük vb.) engeller.
    
    Returns:
        True eğer homografi geçerli, False eğer saçma bir dönüşüm varsa.
    """
    if H is None:
        return False
    
    h, w = image_shape[:2]
    
    # Orijinal köşeleri dönüştür
    corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(corners, H)
    
    # 1. Alan kontrolü: Dönüştürülmüş alan orijinalin %10'undan küçük veya 10 katından büyükse reddet
    original_area = w * h
    transformed_area = cv2.contourArea(transformed.reshape(-1, 2))
    
    if transformed_area < original_area * 0.1 or transformed_area > original_area * 5:
        return False
    
    # 2. Konvekslik kontrolü: Dönüştürülmüş dörtgenin konveks olması lazım (ters dönmemiş)
    if not cv2.isContourConvex(transformed.reshape(-1, 2).astype(np.int32)):
        return False
    
    # 3. Determinant kontrolü: Ayna görüntüsü oluşmamalı
    det = np.linalg.det(H[:2, :2])
    if det < 0.05:
        return False
    
    return True
