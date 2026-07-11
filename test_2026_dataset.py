"""
Teknofest 2026 Veriseti Test Betiği

Referans görselleri yükler, video karelerini işler ve
Görev 3 (Görüntü Eşleme) modülünü test eder.
Çıktı olarak tespit edilen objelerin işaretlendiği bir video üretir.
"""

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
    files = sorted([
        f for f in os.listdir(ref_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    for idx, filename in enumerate(files):
        filepath = os.path.join(ref_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            # ID'yi dosya isminden çıkarmaya çalış (Referans_Nesne_01 -> 1)
            obj_id = idx + 1
            try:
                parts = filename.split('_')
                for part in parts:
                    clean = part.split('.')[0]
                    if clean.isdigit():
                        obj_id = int(clean)
                        break
            except Exception:
                pass

            reference_objects.append({"id": obj_id, "image": img})
            log.info(f"Referans yüklendi: ID={obj_id}, Dosya={filename}, "
                     f"Boyut={img.shape[1]}x{img.shape[0]}")
        else:
            log.warning(f"Görsel okunamadı: {filepath}")

    return reference_objects


def run_dataset_test(video_path, ref_dir, output_path):
    log.info("=" * 60)
    log.info("Teknofest 2026 Veriseti Testi Başlıyor")
    log.info("=" * 60)
    log.info(f"Video: {video_path}")
    log.info(f"Referans Klasörü: {ref_dir}")
    log.info(f"Çıktı: {output_path}")

    if not os.path.exists(video_path):
        log.error("Video bulunamadı!")
        return

    reference_objects = load_reference_objects(ref_dir)
    if not reference_objects:
        log.error("Hiç referans obje yüklenemedi!")
        return

    log.info(f"Toplam {len(reference_objects)} referans obje yüklendi.")

    # ImageMatcher başlat
    matcher = ImageMatcher()

    # Video aç
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.error("Video açılamadı!")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    log.info(f"Video: {width}x{height} @ {fps}FPS, {total_frames} kare")

    # Çıktı video yazıcı
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        log.error("Çıktı video dosyası oluşturulamadı!")
        cap.release()
        return

    frame_count = 0
    total_processing_time = 0

    # İstatistik sayaçları
    detection_counts = {}  # obj_id -> kaç karede tespit edildi
    for ref_obj in reference_objects:
        detection_counts[ref_obj["id"]] = 0

    # Her ID için farklı renk paleti (daha görünür renkler)
    colors = [
        (0, 255, 0),     # Yeşil
        (255, 100, 0),   # Mavi-turuncu
        (0, 0, 255),     # Kırmızı
        (255, 255, 0),   # Cyan
        (0, 255, 255),   # Sarı
        (255, 0, 255),   # Magenta
        (128, 255, 0),   # Açık yeşil
        (0, 128, 255),   # Turuncu
    ]

    try:
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

                # İstatistik güncelle
                if obj_id in detection_counts:
                    detection_counts[obj_id] += 1

                # Her ID için farklı renk
                color = colors[obj_id % len(colors)]

                # Kalın dikdörtgen
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Etiket arka planı
                label = f"ID:{obj_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                label_w, label_h = label_size
                label_y = max(label_h + 10, y1 - 10)
                cv2.rectangle(frame, (x1, label_y - label_h - 6),
                              (x1 + label_w + 6, label_y + 4), color, -1)
                cv2.putText(frame, label, (x1 + 3, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # Sol üst köşe: genel durum bilgisi
            avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
            info_lines = [
                f"Kare: {frame_count}/{total_frames}",
                f"FPS: {current_fps:.1f} (ort: {avg_fps:.1f})",
                f"Bulunan: {len(detections)}/{len(reference_objects)}",
            ]
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 30
                # Gölge efekti
                cv2.putText(frame, line, (22, y_pos + 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(frame, line, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Frame yaz
            out.write(frame)

            # Her 30 karede bir log
            if frame_count % 30 == 0:
                log.info(
                    f"Kare: {frame_count}/{total_frames} | "
                    f"FPS: {avg_fps:.1f} | "
                    f"Bu kare tespitler: {len(detections)}"
                )

    except Exception as e:
        log.error(f"Video işleme sırasında hata: {e}", exc_info=True)
    finally:
        cap.release()
        out.release()

    # Final istatistikler
    log.info("=" * 60)
    log.info("TEST TAMAMLANDI — SONUÇ İSTATİSTİKLERİ")
    log.info("=" * 60)
    log.info(f"İşlenen kare: {frame_count}/{total_frames}")
    if total_processing_time > 0:
        log.info(f"Ortalama FPS: {frame_count / total_processing_time:.1f}")

    log.info("Obje bazında tespit istatistikleri:")
    for obj_id, count in sorted(detection_counts.items()):
        pct = (count / frame_count * 100) if frame_count > 0 else 0
        log.info(f"  Obje {obj_id}: {count}/{frame_count} karede tespit ({pct:.1f}%)")

    log.info(f"Çıktı video: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Teknofest 2026 Görüntü Eşleme (Görev 3) Test Betiği"
    )
    parser.add_argument("--video", type=str, required=True,
                        help="Test edilecek video dosyası")
    parser.add_argument("--ref_dir", type=str, required=True,
                        help="Referans görsellerin bulunduğu klasör")
    parser.add_argument("--out", type=str,
                        default="test_outputs/dataset_sonuc.mp4",
                        help="Çıktı video dosyası")

    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    run_dataset_test(args.video, args.ref_dir, args.out)
