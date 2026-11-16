"""
Görev 3: Görüntü Eşleme — DINOv2 + LoFTR Hibrit Mimari

Pipeline:
  1. DINOv2 ile Kaba Konum Tespiti (Coarse Localization)
     - Referans görselin anlamsal parmak izini (patch token ortalaması) çıkarır
     - Video karesini yama (patch) bazlı tarayarak en benzer aday merkezleri bulur
     - Çok ölçekli (multi-scale) aday kırpma bölgeleri oluşturur

  2. LoFTR ile Hassas Eşleştirme (Fine Matching)
     - Her aday bölgede yoğun piksel eşleştirmesi yapar
     - Homografi + RANSAC + verify_homography ile geometrik doğrulama uygular
     - Hatalı eşleşmeleri reddederek yalnızca doğrulanmış bounding box üretir

  3. CSRT Tracker ile Kare-Arası Takip
     - Tespit edilen nesneleri sonraki karelerde hızlı biçimde takip eder
     - Belirlenen aralıklarla (TRACKER_REDETECTION_INTERVAL) yeniden doğrular
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
from config import (
    DINOV2_MODEL_NAME, DINOV2_SIMILARITY_THRESHOLD,
    DINOV2_INPUT_SIZE, DINOV2_TOP_K_CANDIDATES,
    LOFTR_PRETRAINED, LOFTR_MIN_MATCHES,
    LOFTR_CONFIDENCE_THRESHOLD, LOFTR_MATCH_CONFIDENCE,
    LOFTR_MAX_IMAGE_SIZE,
    CANDIDATE_SCALES, CANDIDATE_MIN_SIZE,
    MOBILESAM_MODEL_PATH, USE_SAM_REFINEMENT,
    TRACKER_REDETECTION_INTERVAL
)
from utils.helper import format_undefined_object_output
from utils.logger import log
from utils.preprocessing import enhance_image, verify_homography


class ImageMatcher:
    """
    Görüntü Eşleme modülü — DINOv2 + LoFTR Hibrit Mimari.

    Yarışma arayüzü (TEAM_INTERFACE) ile tam uyumlu:
      - __init__(device=None): Model yükleme
      - run(frame, reference_objects): Her kare için çağrılır, JSON-uyumlu list döner
    """

    def __init__(self, device=None):
        log.info("Görev 3 (Görüntü Eşleme) modülü başlatılıyor...")
        if device is not None:
            resolved_device = "cpu" if (device == "cuda" and not torch.cuda.is_available()) else device
            self.device = torch.device(resolved_device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Cihaz: {self.device}")

        # Takipçi (Tracker) durumları ve önbellek
        self.trackers = {}
        self.tracker_bboxes = {}          # Son bilinen bbox (tracker sonucu)
        self.frame_count = 0
        self.ref_embeddings_cache = {}    # DINOv2 referans embedding'leri
        self.ref_enhanced_cache = {}      # CLAHE uygulanmış referans görseller
        self.dinov2_patch_size = 14

        # İstatistik sayaçları (loglama için)
        self.stats = {
            "dinov2_candidates": 0,
            "loftr_verified": 0,
            "tracker_used": 0,
            "no_detection": 0,
        }

        # --- DINOv2 Yükleme ---
        self.dinov2 = None
        try:
            self.dinov2 = torch.hub.load(
                'facebookresearch/dinov2', DINOV2_MODEL_NAME,
                pretrained=True
            )
            self.dinov2 = self.dinov2.eval().to(self.device)
            log.info(f"DINOv2 ({DINOV2_MODEL_NAME}) yüklendi.")
        except Exception as e:
            log.error(f"DINOv2 yüklenemedi: {e}")

        # --- LoFTR Yükleme ---
        self.loftr = None
        try:
            from kornia.feature import LoFTR as KorniaLoFTR
            self.loftr = KorniaLoFTR(pretrained=LOFTR_PRETRAINED).eval().to(self.device)
            log.info(f"LoFTR ({LOFTR_PRETRAINED}) yüklendi.")
        except ImportError:
            log.error("kornia bulunamadı! 'pip install kornia' gerekli.")
        except Exception as e:
            log.error(f"LoFTR yüklenemedi: {e}")

        # --- MobileSAM (opsiyonel) ---
        self.use_sam = False
        if USE_SAM_REFINEMENT and os.path.exists(MOBILESAM_MODEL_PATH):
            try:
                from mobile_sam import sam_model_registry, SamPredictor
                sam = sam_model_registry["vit_t"](checkpoint=MOBILESAM_MODEL_PATH)
                sam = sam.eval().to(self.device)
                self.sam_predictor = SamPredictor(sam)
                self.use_sam = True
                log.info("MobileSAM yüklendi. Kutu iyileştirme aktif.")
            except Exception as e:
                log.warning(f"MobileSAM yüklenirken hata: {e}")

        log.info("Görev 3 modülü hazır.")

    # =========================================================================
    # ANA ÇALIŞMA DÖNGÜSÜ
    # =========================================================================
    def run(self, frame, reference_objects=None, **kwargs):
        """
        Her frame için çağrılır.

        Args:
            frame: Ana görüntü (numpy array, BGR).
            reference_objects: [{"id": int, "image": numpy_array}, ...] formatında.

        Returns:
            Eşleşen objelerin listesi (şartname JSON formatında).
        """
        matched_objects = []
        self.frame_count += 1

        if not reference_objects:
            return matched_objects

        # Frame'e CLAHE uygula (DINOv2 + LoFTR'ın detayları görmesi için)
        frame_enhanced = enhance_image(frame, apply_clahe=True, apply_denoise=False)

        # DINOv2 frame özelliklerini tüm objeler için sadece 1 kez çıkar
        frame_patch_data = None

        for ref_obj in reference_objects:
            obj_id = ref_obj["id"]

            try:
                # Offset: Tüm objeler aynı karede aranmasın (hız optimizasyonu)
                try:
                    offset = int(obj_id) * 3
                except (ValueError, TypeError):
                    offset = sum(ord(c) for c in str(obj_id)) * 3

                needs_detection = (
                    obj_id not in self.trackers or
                    (self.frame_count + offset) % TRACKER_REDETECTION_INTERVAL == 0
                )

                if needs_detection:
                    # DINOv2 frame özelliklerini lazy-init
                    if frame_patch_data is None:
                        frame_patch_data = self._extract_frame_patches(frame_enhanced)

                    bbox = self._detect(frame_enhanced, ref_obj, frame_patch_data)

                    if bbox is not None:
                        self._init_tracker(obj_id, frame, bbox)
                        matched_objects.append(self._format_bbox(obj_id, bbox))
                        self.stats["loftr_verified"] += 1
                    else:
                        # Tespit başarısız — tracker varsa onu kullanmayı dene
                        if obj_id in self.trackers:
                            tracked_bbox = self._track(obj_id, frame)
                            if tracked_bbox is not None:
                                matched_objects.append(self._format_bbox(obj_id, tracked_bbox))
                                self.stats["tracker_used"] += 1
                            else:
                                self.trackers.pop(obj_id, None)
                                self.tracker_bboxes.pop(obj_id, None)
                                self.stats["no_detection"] += 1
                        else:
                            self.stats["no_detection"] += 1
                else:
                    # Tracker ile takip et
                    tracked_bbox = self._track(obj_id, frame)
                    if tracked_bbox is not None:
                        matched_objects.append(self._format_bbox(obj_id, tracked_bbox))
                        self.stats["tracker_used"] += 1
                    else:
                        # Tracker başarısız — hemen yeniden tespit dene
                        if frame_patch_data is None:
                            frame_patch_data = self._extract_frame_patches(frame_enhanced)

                        bbox = self._detect(frame_enhanced, ref_obj, frame_patch_data)
                        if bbox is not None:
                            self._init_tracker(obj_id, frame, bbox)
                            matched_objects.append(self._format_bbox(obj_id, bbox))
                            self.stats["loftr_verified"] += 1
                        else:
                            self.trackers.pop(obj_id, None)
                            self.tracker_bboxes.pop(obj_id, None)
                            self.stats["no_detection"] += 1

            except Exception as e:
                log.error(f"Görev 3 Obje {obj_id} çalışırken hata: {e}", exc_info=True)

        # Her 100 karede istatistik logla
        if self.frame_count % 100 == 0:
            log.info(
                f"[Görev 3 İstatistik] Kare {self.frame_count}: "
                f"LoFTR={self.stats['loftr_verified']}, "
                f"Tracker={self.stats['tracker_used']}, "
                f"Boş={self.stats['no_detection']}"
            )

        return matched_objects

    # =========================================================================
    # TESPİT — DINOv2 (Kaba) + LoFTR (İnce) Pipeline
    # =========================================================================
    def _detect(self, frame, ref_obj, frame_patch_data):
        """
        Aşama 1: DINOv2 ile kaba konum tespiti
        Aşama 2: LoFTR ile hassas eşleştirme ve BBox çıkarma

        ÖNEMLİ: LoFTR doğrulaması başarısız olursa None döndürür.
        Yanlış pozitif göndermek, hiç göndermemekten daha kötüdür.
        """
        obj_id = ref_obj["id"]
        ref_img = ref_obj["image"]

        # Referans görseli cache'le ve CLAHE uygula
        if obj_id not in self.ref_enhanced_cache:
            self.ref_enhanced_cache[obj_id] = enhance_image(
                ref_img, apply_clahe=True, apply_denoise=False
            )
        ref_enhanced = self.ref_enhanced_cache[obj_id]

        # === Aşama 1: DINOv2 Kaba Arama ===
        candidate_centers = self._dinov2_coarse_search(
            frame, ref_enhanced, obj_id, frame_patch_data
        )

        if not candidate_centers:
            log.debug(f"Obje {obj_id}: DINOv2 hiç aday bulamadı.")
            return None

        self.stats["dinov2_candidates"] += len(candidate_centers)

        # === Aşama 2: LoFTR İnce Eşleştirme (Multi-Scale) ===
        if self.loftr is None:
            log.warning("LoFTR yüklü değil, tespit yapılamıyor.")
            return None

        best_bbox = None
        best_score = 0
        frame_h, frame_w = frame.shape[:2]
        ref_h, ref_w = ref_enhanced.shape[:2]

        for center_x, center_y, similarity_score in candidate_centers:
            # Multi-scale: Her aday merkezi etrafında farklı boyutlarda kırpma dene
            for scale in CANDIDATE_SCALES:
                # Bölge boyutunu referans boyutuna ve ölçeğe göre hesapla
                # Dron yüksekliğine göre referans obje çok farklı boyutlarda olabilir
                region_w = max(int(ref_w * scale * 1.5), CANDIDATE_MIN_SIZE)
                region_h = max(int(ref_h * scale * 1.5), CANDIDATE_MIN_SIZE)

                # Kare en-boy oranını da dikkate al
                region_w = max(region_w, int(region_h * (ref_w / max(ref_h, 1))))
                region_h = max(region_h, int(region_w * (ref_h / max(ref_w, 1))))

                x1 = max(0, int(center_x - region_w / 2))
                y1 = max(0, int(center_y - region_h / 2))
                x2 = min(frame_w, int(center_x + region_w / 2))
                y2 = min(frame_h, int(center_y + region_h / 2))

                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue

                region_bbox = (x1, y1, x2, y2)
                bbox, score = self._loftr_fine_match(frame, ref_enhanced, region_bbox)

                if bbox is not None and score > best_score:
                    best_bbox = bbox
                    best_score = score

        if best_bbox is not None and best_score >= LOFTR_CONFIDENCE_THRESHOLD:
            log.info(
                f"Obje {obj_id}: LoFTR DOĞRULADI — skor {best_score:.3f}, "
                f"BBox ({best_bbox[0]},{best_bbox[1]})-({best_bbox[2]},{best_bbox[3]})"
            )
            return best_bbox

        # LoFTR doğrulayamadı — hiçbir şey döndürme
        log.debug(
            f"Obje {obj_id}: LoFTR doğrulayamadı "
            f"(en iyi skor: {best_score:.3f}, eşik: {LOFTR_CONFIDENCE_THRESHOLD}). "
            f"Tespit atlanıyor."
        )
        return None

    # =========================================================================
    # AŞAMA 1: DINOv2 KABA KONUM TESPİTİ
    # =========================================================================
    def _extract_frame_patches(self, frame):
        """
        Frame'in DINOv2 patch token'larını çıkarır.
        Bu fonksiyon her kare için SADECE 1 KEZ çağrılır (tüm objeler için paylaşılır).
        """
        if self.dinov2 is None:
            return None

        input_size = DINOV2_INPUT_SIZE
        frame_resized = cv2.resize(frame, (input_size, input_size))
        frame_tensor = self._prepare_dinov2_input(frame_resized)

        with torch.inference_mode():
            features = self.dinov2.forward_features(frame_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # [1, N, D]

        num_patches = input_size // self.dinov2_patch_size
        return {
            "patch_tokens": patch_tokens.squeeze(0),  # [N, D]
            "num_patches": num_patches,
            "frame_shape": frame.shape[:2]  # (H, W)
        }

    def _dinov2_coarse_search(self, frame, ref_img, obj_id, frame_patch_data):
        """
        DINOv2 patch-token tabanlı kaba arama.

        Referans görselin patch token ortalamasını alıp frame patch'leriyle
        cosine similarity hesaplar. En yüksek benzerlikli merkez noktalarını döndürür.

        Returns:
            List of (center_x, center_y, similarity_score) tuples.
        """
        if frame_patch_data is None or self.dinov2 is None:
            return []

        # Referans embedding'i cache'le (her referans için sadece 1 kez)
        if obj_id not in self.ref_embeddings_cache:
            ref_emb = self._extract_dinov2_embedding(ref_img)
            if ref_emb is None:
                return []
            self.ref_embeddings_cache[obj_id] = ref_emb

        ref_emb = self.ref_embeddings_cache[obj_id]  # [D]
        patch_tokens = frame_patch_data["patch_tokens"]  # [N, D]
        num_patches = frame_patch_data["num_patches"]
        frame_h, frame_w = frame_patch_data["frame_shape"]

        # Cosine similarity heatmap
        similarity = F.cosine_similarity(
            patch_tokens,              # [N, D]
            ref_emb.unsqueeze(0),      # [1, D]
            dim=1
        )  # [N]

        heatmap = similarity.cpu().numpy().reshape(num_patches, num_patches)

        # Heatmap'i yumuşat (birbirine yakın yüksek değerleri birleştir)
        from scipy.ndimage import gaussian_filter
        heatmap_smooth = gaussian_filter(heatmap, sigma=1.5)

        # Dinamik eşik: ortalama + 1 standart sapma veya sabit eşikten yüksek olanı
        dynamic_threshold = max(
            DINOV2_SIMILARITY_THRESHOLD,
            float(heatmap_smooth.mean() + 1.0 * heatmap_smooth.std())
        )

        candidates = []
        heatmap_work = heatmap_smooth.copy()

        for _ in range(DINOV2_TOP_K_CANDIDATES):
            max_val = heatmap_work.max()
            if max_val < dynamic_threshold:
                break

            max_idx = np.unravel_index(heatmap_work.argmax(), heatmap_work.shape)
            py, px = max_idx

            # Patch koordinatlarını orijinal frame piksellerine çevir
            patch_w = frame_w / num_patches
            patch_h = frame_h / num_patches

            center_x = (px + 0.5) * patch_w
            center_y = (py + 0.5) * patch_h

            candidates.append((center_x, center_y, float(max_val)))

            # Bu bölgeyi maskele (aynı bölge tekrar seçilmesin)
            suppress_r = 3
            py_min = max(0, py - suppress_r)
            py_max = min(num_patches, py + suppress_r + 1)
            px_min = max(0, px - suppress_r)
            px_max = min(num_patches, px + suppress_r + 1)
            heatmap_work[py_min:py_max, px_min:px_max] = 0

        if candidates:
            log.debug(
                f"Obje {obj_id}: DINOv2 {len(candidates)} aday buldu "
                f"(eşik: {dynamic_threshold:.3f}, max: {heatmap_smooth.max():.3f})"
            )

        return candidates

    # =========================================================================
    # AŞAMA 2: LoFTR HASSAS EŞLEŞTİRME
    # =========================================================================
    def _loftr_fine_match(self, frame, ref_img, candidate_bbox):
        """
        LoFTR ile aday bölge ve referans arasında yoğun eşleştirme.
        Homografi + RANSAC ile doğrulama, verify_homography ile geometrik kontrol,
        hassas BBox çıkarma.

        Returns:
            (bbox_tuple, score) veya (None, 0)
        """
        x1, y1, x2, y2 = [int(v) for v in candidate_bbox]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 30 or crop.shape[1] < 30:
            return None, 0

        # Gri tonlamaya çevir (LoFTR grayscale çalışır)
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        # Boyutları LoFTR için uygun hale getir (8'in katı, max boyut sınırı)
        ref_resized, ref_scale = self._resize_for_loftr(ref_gray)
        crop_resized, crop_scale = self._resize_for_loftr(crop_gray)

        if ref_resized is None or crop_resized is None:
            return None, 0

        # Tensor'e çevir: [B, 1, H, W]
        img0 = torch.from_numpy(ref_resized).float()[None, None].to(self.device) / 255.0
        img1 = torch.from_numpy(crop_resized).float()[None, None].to(self.device) / 255.0

        with torch.inference_mode():
            correspondences = self.loftr({"image0": img0, "image1": img1})

        mkpts0 = correspondences["keypoints0"].cpu().numpy()
        mkpts1 = correspondences["keypoints1"].cpu().numpy()
        conf = correspondences["confidence"].cpu().numpy()

        # Güvenilir eşleşmeleri filtrele
        high_conf_mask = conf > LOFTR_MATCH_CONFIDENCE
        mkpts0 = mkpts0[high_conf_mask]
        mkpts1 = mkpts1[high_conf_mask]

        if len(mkpts0) < LOFTR_MIN_MATCHES:
            return None, 0

        # Homografi hesapla (RANSAC ile)
        H, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
        if H is None or inlier_mask is None:
            return None, 0

        num_inliers = int(np.sum(inlier_mask))
        if num_inliers < LOFTR_MIN_MATCHES:
            return None, 0

        # Geometrik doğrulama: verify_homography ile saçma kutuları reddet
        h0, w0 = ref_resized.shape[:2]
        if not verify_homography(H, (h0, w0)):
            return None, 0

        # Referans köşelerini aday bölgeye yansıt
        corners = np.float32([
            [0, 0], [0, h0 - 1], [w0 - 1, h0 - 1], [w0 - 1, 0]
        ]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)
        pts = transformed.reshape(-1, 2)

        # Koordinat dönüşümü: crop_resized → crop → frame
        crop_h, crop_w = crop.shape[:2]
        resized_h, resized_w = crop_resized.shape[:2]

        pts[:, 0] = pts[:, 0] * (crop_w / resized_w) + x1
        pts[:, 1] = pts[:, 1] * (crop_h / resized_h) + y1

        # BBox hesapla
        bx1 = max(0, int(pts[:, 0].min()))
        by1 = max(0, int(pts[:, 1].min()))
        bx2 = min(frame.shape[1], int(pts[:, 0].max()))
        by2 = min(frame.shape[0], int(pts[:, 1].max()))

        if bx2 <= bx1 or by2 <= by1:
            return None, 0

        # Ek geometrik doğrulama
        bbox_w = bx2 - bx1
        bbox_h = by2 - by1
        bbox_area = bbox_w * bbox_h
        frame_area = frame.shape[0] * frame.shape[1]

        # Çok büyük kutular (frame'in %50'sinden fazla) veya çok küçük kutular reddet
        if bbox_area > frame_area * 0.5 or bbox_area < 200:
            return None, 0

        # En-boy oranı kontrolü: aşırı uzun/ince kutular reddet
        aspect_ratio = max(bbox_w, bbox_h) / max(min(bbox_w, bbox_h), 1)
        if aspect_ratio > 10:
            return None, 0

        # Güven skoru = inlier oranı
        score = num_inliers / max(len(mkpts0), 1)

        return (bx1, by1, bx2, by2), score

    # =========================================================================
    # DINOv2 YARDIMCI FONKSİYONLAR
    # =========================================================================
    def _extract_dinov2_embedding(self, image):
        """
        Görüntüden DINOv2 patch token ortalaması çıkarır.

        CLS token yerine patch token ortalaması kullanıyoruz çünkü:
        - CLS token tüm görüntünün "özetini" verir (arka plan dahil)
        - Patch token ortalaması nesnenin dokusuna daha çok odaklanır
        - Referans görsellerde arka plan az olduğu için fark küçük,
          ama video karelerinde arka plan baskın olduğu için bu önemli
        """
        if self.dinov2 is None:
            return None

        input_size = DINOV2_INPUT_SIZE
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = self._prepare_dinov2_input(img_resized)

        with torch.inference_mode():
            features = self.dinov2.forward_features(img_tensor)
            patch_tokens = features["x_norm_patchtokens"]  # [1, N, D]

        # Patch token ortalaması (CLS yerine — nesneye daha çok odaklanır)
        return patch_tokens.squeeze(0).mean(dim=0)  # [D]

    def _prepare_dinov2_input(self, image):
        """OpenCV BGR → DINOv2 normalized tensor."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img = image.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
        return tensor.to(self.device)

    # =========================================================================
    # LoFTR YARDIMCI FONKSİYONLAR
    # =========================================================================
    def _resize_for_loftr(self, gray_img):
        """Görüntüyü LoFTR için uygun boyuta getirir (8'in katı, max boyut sınırı)."""
        h, w = gray_img.shape[:2]
        if h < 16 or w < 16:
            return None, 1.0

        # Max boyut sınırı
        max_dim = LOFTR_MAX_IMAGE_SIZE
        scale = min(max_dim / max(h, w), 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 8'in katına yuvarla
        new_w = max(16, (new_w // 8) * 8)
        new_h = max(16, (new_h // 8) * 8)

        resized = cv2.resize(gray_img, (new_w, new_h))
        return resized, scale

    # =========================================================================
    # TRACKER
    # =========================================================================
    def _init_tracker(self, obj_id, frame, bbox):
        """CSRT tracker başlatır."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return
        tracker = cv2.TrackerCSRT_create()
        tracker.init(frame, (x1, y1, w, h))
        self.trackers[obj_id] = tracker
        self.tracker_bboxes[obj_id] = bbox

    def _track(self, obj_id, frame):
        """Mevcut tracker ile objeyi takip eder."""
        if obj_id not in self.trackers:
            return None
        success, box = self.trackers[obj_id].update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            bbox = (x, y, x + w, y + h)
            self.tracker_bboxes[obj_id] = bbox
            return bbox
        return None

    # =========================================================================
    # FORMAT
    # =========================================================================
    def _format_bbox(self, obj_id, bbox):
        """BBox'ı şartname JSON formatına çevirir."""
        return format_undefined_object_output(
            obj_id=obj_id,
            top_left_x=float(bbox[0]),
            top_left_y=float(bbox[1]),
            bottom_right_x=float(bbox[2]),
            bottom_right_y=float(bbox[3])
        )
