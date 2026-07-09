"""
Görev 3: Görüntü Eşleme — DINOv2 + LoFTR Hibrit Mimari

Pipeline:
  1. DINOv2 ile kaba konum tespiti (Coarse Localization)
     - Referans görselin "anlamsal parmak izi"ni (CLS embedding) çıkar
     - Video karesini yama (patch) bazlı tara, en benzer bölgeleri bul
     - Bu aşama bakış açısı, ölçek ve hatta modalite (termal/RGB) farkına dayanıklıdır
  
  2. LoFTR ile hassas eşleştirme (Fine Matching)
     - DINOv2'nin bulduğu aday bölgelerde yoğun piksel eşleştirmesi yap
     - Homografi + RANSAC ile geometrik doğrulama
     - Hassas bounding box çıkar
  
  3. CSRT Tracker ile kare-arası takip
     - Obje bulunduğunda, sonraki karelerde hızlıca takip et
     - Periyodik olarak DINOv2+LoFTR ile tekrar doğrula

Neden bu mimari?
  - LightGlue/SuperPoint BAŞARISIZ oldu çünkü seyrek öznitelik eşleştirici,
    büyük bakış açısı farkları (yandan çekim → üstten drone) ve dokusuz 
    yüzeyler (çim, metal) ile baş edemiyor.
  - DINOv2 kavramsal (semantic) seviyede çalışır: "bu bir biçerdöver" gibi
    üst düzey anlayışa sahiptir, piksel düzeyinde eşleştirme gerektirmez.
  - LoFTR dedektör-bağımsız (detector-free) yoğun eşleştirme yapar,
    SuperPoint'in aksine nokta bulmaya ihtiyaç duymaz.
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
    LOFTR_CONFIDENCE_THRESHOLD, LOFTR_MAX_IMAGE_SIZE,
    MOBILESAM_MODEL_PATH, USE_SAM_REFINEMENT,
    TRACKER_REDETECTION_INTERVAL
)
from utils.helper import format_undefined_object_output
from utils.logger import log
from utils.preprocessing import enhance_image


class ImageMatcher:
    """
    Görüntü Eşleme modülü — DINOv2 + LoFTR Hibrit Mimari.
    
    Yarışma arayüzü (TEAM_INTERFACE) ile tam uyumlu:
      - __init__(): Model yükleme
      - run(frame, reference_objects): Her kare için çağrılır, JSON-uyumlu list döner
    """

    def __init__(self):
        log.info("Görev 3 (Görüntü Eşleme) modülü başlatılıyor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Cihaz: {self.device}")

        # Takipçi (Tracker) durumları ve önbellek
        self.trackers = {}
        self.frame_count = 0
        self.ref_embeddings_cache = {}   # DINOv2 referans embedding'leri
        self.dinov2_patch_size = 14

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

        # DINOv2 frame özelliklerini tüm objeler için sadece 1 kez çıkar
        frame_patch_data = None

        try:
            for ref_obj in reference_objects:
                obj_id = ref_obj["id"]

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
                    # DINOv2 frame özelliklerini lazy-init (ilk ihtiyaç duyulduğunda)
                    if frame_patch_data is None:
                        frame_patch_data = self._extract_frame_patches(frame)

                    bbox = self._detect(frame, ref_obj, frame_patch_data)

                    if bbox is not None:
                        self._init_tracker(obj_id, frame, bbox)
                        matched_objects.append(self._format_bbox(obj_id, bbox))
                        log.info(f"Obje {obj_id} tespit edildi. BBox: {bbox}")
                    else:
                        log.debug(f"Obje {obj_id} bu karede bulunamadı.")
                        self.trackers.pop(obj_id, None)
                else:
                    # Tracker ile takip et
                    tracked_bbox = self._track(obj_id, frame)
                    if tracked_bbox is not None:
                        matched_objects.append(self._format_bbox(obj_id, tracked_bbox))
                    else:
                        log.warning(f"Obje {obj_id} takibi kaybedildi.")

        except Exception as e:
            log.error(f"Görev 3 çalışırken hata: {e}", exc_info=True)

        return matched_objects

    # =========================================================================
    # TESPİT — DINOv2 (Kaba) + LoFTR (İnce) Pipeline
    # =========================================================================
    def _detect(self, frame, ref_obj, frame_patch_data):
        """
        Aşama 1: DINOv2 ile kaba konum tespiti
        Aşama 2: LoFTR ile hassas eşleştirme ve BBox çıkarma
        """
        obj_id = ref_obj["id"]
        ref_img = ref_obj["image"]

        # === Aşama 1: DINOv2 Kaba Arama ===
        candidate_regions = self._dinov2_coarse_search(
            frame, ref_img, obj_id, frame_patch_data
        )

        if not candidate_regions:
            return None

        # === Aşama 2: LoFTR İnce Eşleştirme ===
        if self.loftr is not None:
            best_bbox = None
            best_score = 0

            for region_bbox in candidate_regions:
                bbox, score = self._loftr_fine_match(frame, ref_img, region_bbox)
                if bbox is not None and score > best_score:
                    best_bbox = bbox
                    best_score = score

            if best_bbox is not None and best_score >= LOFTR_CONFIDENCE_THRESHOLD:
                log.debug(f"Obje {obj_id}: LoFTR eşleştirme skoru {best_score:.2f}")
                return best_bbox

        # LoFTR başarısız veya yüklü değilse, DINOv2'nin en iyi adayını döndür
        # (Bu fallback, en azından bir kutu üretir)
        log.debug(f"Obje {obj_id}: LoFTR eşleşme bulamadı, DINOv2 aday bölgesi kullanılıyor.")
        return candidate_regions[0]

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
        
        Her yama (patch) için referans ile anlamsal benzerlik hesaplar.
        En yüksek benzerlikli bölgeleri aday olarak döndürür.
        """
        if frame_patch_data is None or self.dinov2 is None:
            return []

        # Referans embedding'i cache'le (her referans için sadece 1 kez)
        if obj_id not in self.ref_embeddings_cache:
            ref_emb = self._extract_dinov2_cls(ref_img)
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
        heatmap_smooth = gaussian_filter(heatmap, sigma=1.0)

        # Dinamik eşik: ortalama + 1.5 standart sapma veya sabit eşikten yüksek olanı
        dynamic_threshold = max(
            DINOV2_SIMILARITY_THRESHOLD,
            float(heatmap_smooth.mean() + 1.5 * heatmap_smooth.std())
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

            cx = (px + 0.5) * patch_w
            cy = (py + 0.5) * patch_h

            # Aday bölge boyutu: referans objenin en-boy oranına göre ölçekle
            ref_h, ref_w = ref_img.shape[:2]
            ref_aspect = ref_w / max(ref_h, 1)

            # Aday bölgenin boyutunu kareye ve referans boyutuna göre ayarla
            # Obje dron yüksekliğine bağlı olarak çok farklı boyutlarda olabilir
            search_size = max(patch_w * 6, patch_h * 6, 100)

            half_w = search_size * max(ref_aspect, 1.0) / 2
            half_h = search_size / max(ref_aspect, 1.0) / 2
            # Minimum boyut garantisi
            half_w = max(half_w, 60)
            half_h = max(half_h, 60)

            x1 = max(0, int(cx - half_w))
            y1 = max(0, int(cy - half_h))
            x2 = min(frame_w, int(cx + half_w))
            y2 = min(frame_h, int(cy + half_h))

            if (x2 - x1) > 20 and (y2 - y1) > 20:
                candidates.append((x1, y1, x2, y2))

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
        Homografi + RANSAC ile doğrulama, hassas BBox çıkarma.
        """
        x1, y1, x2, y2 = [int(v) for v in candidate_bbox]
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
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
        high_conf_mask = conf > 0.5
        mkpts0 = mkpts0[high_conf_mask]
        mkpts1 = mkpts1[high_conf_mask]

        if len(mkpts0) < LOFTR_MIN_MATCHES:
            return None, 0

        # Homografi hesapla (RANSAC ile)
        H, inlier_mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 3.0)
        if H is None or inlier_mask is None:
            return None, 0

        num_inliers = int(np.sum(inlier_mask))
        if num_inliers < LOFTR_MIN_MATCHES:
            return None, 0

        # Referans köşelerini aday bölgeye yansıt
        h0, w0 = ref_resized.shape[:2]
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

        # Geometrik doğrulama: saçma kutular reddedilir
        bbox_area = (bx2 - bx1) * (by2 - by1)
        frame_area = frame.shape[0] * frame.shape[1]
        if bbox_area > frame_area * 0.5 or bbox_area < 100:
            return None, 0

        # Güven skoru = inlier oranı
        score = num_inliers / max(len(mkpts0), 1)

        return (bx1, by1, bx2, by2), score

    # =========================================================================
    # DINOv2 YARDIMCI FONKSİYONLAR
    # =========================================================================
    def _extract_dinov2_cls(self, image):
        """Görüntüden DINOv2 CLS token'ı çıkarır (anlamsal parmak izi)."""
        if self.dinov2 is None:
            return None

        input_size = DINOV2_INPUT_SIZE
        img_resized = cv2.resize(image, (input_size, input_size))
        img_tensor = self._prepare_dinov2_input(img_resized)

        with torch.inference_mode():
            features = self.dinov2.forward_features(img_tensor)
            cls_token = features["x_norm_clstoken"]  # [1, D]

        return cls_token.squeeze(0)  # [D]

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
        if h < 8 or w < 8:
            return None, 1.0

        # Max boyut sınırı
        max_dim = LOFTR_MAX_IMAGE_SIZE
        scale = min(max_dim / max(h, w), 1.0)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # 8'in katına yuvarla
        new_w = max(8, (new_w // 8) * 8)
        new_h = max(8, (new_h // 8) * 8)

        resized = cv2.resize(gray_img, (new_w, new_h))
        return resized, scale

    # =========================================================================
    # TRACKER (Mevcut — değişmedi)
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

    def _track(self, obj_id, frame):
        """Mevcut tracker ile objeyi takip eder."""
        if obj_id not in self.trackers:
            return None
        success, box = self.trackers[obj_id].update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            return (x, y, x + w, y + h)
        return None

    # =========================================================================
    # FORMAT (Mevcut — değişmedi)
    # =========================================================================
    def _format_bbox(self, obj_id, bbox):
        """BBox'ı şartname JSON formatına çevirir."""
        return format_undefined_object_output(
            object_id=obj_id,
            top_left_x=float(bbox[0]),
            top_left_y=float(bbox[1]),
            bottom_right_x=float(bbox[2]),
            bottom_right_y=float(bbox[3])
        )
