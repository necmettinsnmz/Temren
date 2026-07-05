import os
import cv2
import time
import argparse
from tasks.task3_matching import ImageMatcher
from utils.logger import log

def load_reference_objects(ref_dir):
    """
    Verilen klasördeki tüm referans resimleri yükler ve 
    ImageMatcher'ın beklediği formatta döndürür: [{"id": 1, "image": img}, ...]
    """
    reference_objects = []
    
    if not os.path.exists(ref_dir):
        log.error(f"Referans klasörü bulunamadı: {ref_dir}")
        return reference_objects
        
    # Klasördeki görselleri al ve isme göre sırala (Nesne_01, Nesne_02...)
    files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    for idx, filename in enumerate(files):
        filepath = os.path.join(ref_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            # ID'yi dosya isminden çıkarmaya çalış (Referans_Nesne_01 -> 1)
            # Yapamazsa index + 1 kullan
            obj_id = idx + 1
            try:
                # "Referans_Nesne_01.JPG" -> "01" -> 1
                parts = filename.split('_')
                for part in parts:
                    clean = part.split('.')[0]
                    if clean.isdigit():
                        obj_id = int(clean)
                        break
            except:
                pass
                
            reference_objects.append({"id": obj_id, "image": img})
            log.info(f"Referans yüklendi: ID={obj_id}, Dosya={filename}")
        else:
            log.warning(f"Görsel okunamadı: {filepath}")
            
    return reference_objects

def run_dataset_test(video_path, ref_dir, output_path):
    log.info(f"Teknofest 2026 Veriseti Testi Başlıyor")
    log.info(f"Video: {video_path}")
    log.info(f"Referans Klasörü: {ref_dir}")
    
    if not os.path.exists(video_path):
        log.error("Video bulunamadı!")
        return
        
    reference_objects = load_reference_objects(ref_dir)
    if not reference_objects:
        log.error("Hiç referans obje yüklenemedi!")
        return
        
    matcher = ImageMatcher()
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_processing_time = 0
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        start_time = time.time()
        detections = matcher.run(frame, reference_objects=reference_objects)
        elapsed_time = time.time() - start_time
        total_processing_time += elapsed_time
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        
        # Çizimler
        for det in detections:
            obj_id = det["object_id"]
            x1 = int(det["top_left_x"])
            y1 = int(det["top_left_y"])
            x2 = int(det["bottom_right_x"])
            y2 = int(det["bottom_right_y"])
            
            # Her ID için farklı renk
            color = colors[obj_id % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Etiket
            label = f"ID: {obj_id}"
            cv2.putText(frame, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        # Genel FPS ve durum
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Bulunan Obje: {len(detections)}/{len(reference_objects)}", 
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            log.info(f"İşlenen Kare: {frame_count}/{total_frames} - Ortalama FPS: {frame_count/total_processing_time:.1f}")

    cap.release()
    out.release()
    log.info(f"İşlem tamamlandı! Çıktı: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="Test edilecek video dosyası")
    parser.add_argument("--ref_dir", type=str, required=True, help="Referans görsellerin bulunduğu klasör")
    parser.add_argument("--out", type=str, default="test_outputs/dataset_sonuc.mp4", help="Çıktı video dosyası")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    run_dataset_test(args.video, args.ref_dir, args.out)
