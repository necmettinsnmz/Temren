import time
import json
from config import USER_URL
from utils.server_api import fetch_frame_data, send_results
from utils.helper import load_image_from_url
from tasks.task1_detection import ObjectDetector
from tasks.task2_position import PositionEstimator
from tasks.task3_matching import ImageMatcher
from utils.logger import log

def main():
    log.info("="*50)
    log.info("Teknofest 2026 Havacılıkta Yapay Zeka Başlıyor")
    log.info("="*50)

    # 1. Modüllerin Başlatılması
    task1 = ObjectDetector()
    task2 = PositionEstimator()
    task3 = ImageMatcher()

    # Örnek oturum url'i, yarışma anında dinamik olabilir
    session_url = "http://localhost/session/2/"

    # Yarışma esnasında saniyede 7.5 FPS (yaklaşık 133ms) hızında frameler işlenecek
    # Biz burada örnek bir döngü kuruyoruz
    for frame_id in range(1, 6): # 5 karelik test
        log.info(f"\n[DÖNGÜ] Frame {frame_id} işleniyor...")
        start_time = time.time()

        # 2. Veri Çekme
        telemetry = fetch_frame_data(session_url)
        current_frame = load_image_from_url(telemetry["image_url"])

        # 3. Görevleri Çalıştırma
        detected_objects = task1.run(current_frame)
        detected_translations = task2.run(current_frame)
        # Referans objeler sunucudan veya başka yerden gelebilir, şimdilik None geçiyoruz
        detected_undefined_objects = task3.run(current_frame, reference_objects=None)

        # 4. JSON Sonuç Paketini Oluşturma
        payload = {
            "id": f"{telemetry['video_name']}_f{frame_id}",
            "user": USER_URL,
            "frame": telemetry["url"],
            "detected_objects": detected_objects,
            "detected_translations": detected_translations,
            "detected_undefined_objects": detected_undefined_objects
        }

        # 5. Sunucuya Gönderme
        send_results(payload)

        # Gecikme hesaplama (7.5 FPS için)
        elapsed = time.time() - start_time
        sleep_time = max(0, (1.0 / 7.5) - elapsed)
        time.sleep(sleep_time)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"Ana döngüde beklenmeyen hata: {e}", exc_info=True)
