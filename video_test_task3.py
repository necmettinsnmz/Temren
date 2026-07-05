import cv2
import time
import os
from tasks.task3_matching import ImageMatcher
from utils.logger import log

def run_video_test(video_path, ref_image_path, output_path="output_video.mp4"):
    if not os.path.exists(video_path):
        print(f"[HATA] Video dosyası bulunamadı: {video_path}")
        return
    if not os.path.exists(ref_image_path):
        print(f"[HATA] Referans görsel bulunamadı: {ref_image_path}")
        return

    # Modeli Başlat
    print("[BİLGİ] Model yükleniyor...")
    matcher = ImageMatcher()
    
    # Referans objeyi yükle
    ref_img = cv2.imread(ref_image_path)
    ref_objects = [{"id": 1, "image": ref_img}]
    
    # Videoyu Aç
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Çıktı Videosu Ayarları
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"[BİLGİ] Video işleniyor: {width}x{height} @ {fps} FPS, Toplam Kare: {total_frames}")
    
    frame_count = 0
    total_processing_time = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        start_time = time.time()
        # Modeli çalıştır (İlk karelerde LightGlue, sonrasında CSRT Tracker çalışacak)
        detections = matcher.run(frame, reference_objects=ref_objects)
        elapsed_time = time.time() - start_time
        total_processing_time += elapsed_time
        
        # Anlık FPS hesaplama
        current_fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
        
        # Çizim İşlemleri
        if detections:
            det = detections[0]
            x1, y1 = int(det["top_left_x"]), int(det["top_left_y"])
            x2, y2 = int(det["bottom_right_x"]), int(det["bottom_right_y"])
            
            # Takip kutusu
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Model durumunu (LightGlue mu Tracker mı?) yazdırma
            # matcher.frame_count üzerinden hangi fazda olduğumuzu anlarız
            status_text = "LightGlue (Tespit)" if (matcher.frame_count - 1) % 15 == 0 else "CSRT (Takip)"
            cv2.putText(frame, f"Durum: {status_text}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Obje Bulunamadi!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # Sol üste FPS ve bilgi yazdır
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"İşlenen Kare: {frame_count}/{total_frames} - Ortalama FPS: {frame_count/total_processing_time:.1f}")

    cap.release()
    out.release()
    print(f"\n[BİLGİ] İşlem tamamlandı! Video kaydedildi: {output_path}")
    print(f"[BİLGİ] Ortalama FPS: {frame_count/total_processing_time:.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Temren Video Test Scripti")
    parser.add_argument("--video", type=str, required=True, help="Test edilecek MP4 video yolu")
    parser.add_argument("--ref", type=str, required=True, help="Referans obje görseli (örn: ref.jpg)")
    parser.add_argument("--out", type=str, default="output_video.mp4", help="Çıktı video adı")
    
    args = parser.parse_args()
    run_video_test(args.video, args.ref, args.out)
