import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
"""
Görev 3 Benchmark ve Test Scripti

Bu script:
1. İnternetten örnek görüntüler indirir
2. Sentetik referans obje/frame çiftleri oluşturur (crop + transform)  
3. Pipeline'ı çalıştırıp IoU skorunu ölçer
4. FPS performansını raporlar
5. Farklı zorluk seviyelerinde (ölçek, rotasyon, aydınlatma) test eder
"""

import cv2
import numpy as np
import time
import os
import urllib.request
from tasks.task3_matching import ImageMatcher
from utils.logger import log


def download_test_images(output_dir="test_data"):
    """İnternetten test görüntüleri indirir."""
    os.makedirs(output_dir, exist_ok=True)
    
    urls = {
        "scene1.jpg": "https://raw.githubusercontent.com/cvg/LightGlue/main/assets/sacre_coeur1.jpg",
        "scene2.jpg": "https://raw.githubusercontent.com/cvg/LightGlue/main/assets/sacre_coeur2.jpg",
    }
    
    for filename, url in urls.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            log.info(f"İndiriliyor: {filename}")
            urllib.request.urlretrieve(url, filepath)
    
    return output_dir


def calculate_iou(box1, box2):
    """
    İki bounding box arasındaki IoU (Intersection over Union) değerini hesaplar.
    box format: (x1, y1, x2, y2)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def create_synthetic_pairs(image, num_pairs=5):
    """
    Bir görüntüden sentetik referans-frame çiftleri üretir.
    Farklı zorluk seviyeleri: crop, scale, rotation, brightness.
    """
    h, w = image.shape[:2]
    pairs = []

    # Test 1: Basit kırpma (kolay)
    crop_x, crop_y = w // 4, h // 4
    crop_w, crop_h = w // 3, h // 3
    ref = image[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w].copy()
    gt_bbox = (crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
    pairs.append(("Basit Kırpma", ref, image.copy(), gt_bbox))

    # Test 2: Küçültülmüş referans (orta)
    scale = 0.5
    ref_scaled = cv2.resize(ref, None, fx=scale, fy=scale)
    pairs.append(("Ölçek Değişimi (0.5x)", ref_scaled, image.copy(), gt_bbox))

    # Test 3: Döndürülmüş referans (zor)
    center = (crop_w // 2, crop_h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    ref_rotated = cv2.warpAffine(ref, M, (crop_w, crop_h))
    pairs.append(("15° Rotasyon", ref_rotated, image.copy(), gt_bbox))

    # Test 4: Parlaklık değişimi (orta)
    ref_bright = cv2.convertScaleAbs(ref, alpha=1.3, beta=30)
    pairs.append(("Parlaklık Artışı", ref_bright, image.copy(), gt_bbox))

    # Test 5: Kontrast düşürülmüş (zor)
    ref_low_contrast = cv2.convertScaleAbs(ref, alpha=0.6, beta=10)
    pairs.append(("Düşük Kontrast", ref_low_contrast, image.copy(), gt_bbox))

    return pairs


def run_benchmark():
    """Ana benchmark fonksiyonu."""
    print("=" * 70)
    print("  GÖREV 3 - GÖRÜNTÜ EŞLEME BENCHMARK TESTİ")
    print("=" * 70)

    # Test görüntülerini indir
    test_dir = download_test_images()
    
    # Pipeline'ı başlat
    matcher = ImageMatcher()

    # Test görüntüsünü yükle
    scene_path = os.path.join(test_dir, "scene1.jpg")
    scene = cv2.imread(scene_path)
    if scene is None:
        print("[HATA] Test görüntüsü yüklenemedi!")
        return

    print(f"\nGörüntü boyutu: {scene.shape[1]}x{scene.shape[0]}")
    print("-" * 70)

    # Sentetik test çiftleri oluştur
    pairs = create_synthetic_pairs(scene)

    results = []
    total_time = 0

    for test_name, ref_img, frame, gt_bbox in pairs:
        ref_objects = [{"id": 1, "image": ref_img}]
        
        # Matcher'ın frame sayacını sıfırla (her test bağımsız olsun)
        matcher.frame_count = 0
        matcher.trackers = {}

        start = time.time()
        detections = matcher.run(frame, reference_objects=ref_objects)
        elapsed = time.time() - start
        total_time += elapsed

        if detections:
            det = detections[0]
            pred_bbox = (
                det["top_left_x"], det["top_left_y"],
                det["bottom_right_x"], det["bottom_right_y"]
            )
            iou = calculate_iou(gt_bbox, pred_bbox)
            status = "[OK] BASARILI" if iou >= 0.5 else "[!!] DUSUK IoU"
        else:
            iou = 0.0
            pred_bbox = None
            status = "[XX] TESPIT EDILEMEDI"

        results.append({
            "test": test_name,
            "iou": iou,
            "time_ms": elapsed * 1000,
            "status": status,
            "gt": gt_bbox,
            "pred": pred_bbox
        })

        print(f"\n  [{status}] {test_name}")
        print(f"    IoU: {iou:.4f} | Süre: {elapsed*1000:.1f}ms")
        if pred_bbox:
            print(f"    GT:   {gt_bbox}")
            print(f"    Pred: {pred_bbox}")

    # Özet
    print("\n" + "=" * 70)
    print("  ÖZET")
    print("=" * 70)
    
    ious = [r["iou"] for r in results]
    successful = sum(1 for r in results if r["iou"] >= 0.5)
    
    print(f"  Toplam test    : {len(results)}")
    print(f"  Basarili (>=0.5): {successful}/{len(results)}")
    print(f"  Ortalama IoU   : {np.mean(ious):.4f}")
    print(f"  Min IoU        : {np.min(ious):.4f}")
    print(f"  Max IoU        : {np.max(ious):.4f}")
    print(f"  Toplam süre    : {total_time*1000:.1f}ms")
    print(f"  Ort. FPS       : {len(results)/total_time:.1f}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_benchmark()
