import cv2
import numpy as np
import torch
import os
from config import (
    LIGHTGLUE_MAX_KEYPOINTS, LIGHTGLUE_MIN_MATCHES,
    MOBILESAM_MODEL_PATH, USE_SAM_REFINEMENT,
    TRACKER_REDETECTION_INTERVAL
)
from utils.helper import format_undefined_object_output
from utils.logger import log
from utils.preprocessing import (
    enhance_image, compute_match_confidence, verify_homography
)


class ImageMatcher:
    """
    Görüntü Eşleme modülü.
    
    Pipeline:
    1. SuperPoint + LightGlue ile referans objeyi ana karede bul (feature matching).
    2. MobileSAM ile eşleşen bölgenin piksel düzeyinde maskesini çıkar ve kutuyu iyileştir.
    3. CSRT Tracker ile ara karelerde hızlıca takip et, belirli aralıklarla LightGlue ile doğrula.
    """

    def __init__(self):
        log.info("Görev 3 (Görüntü Eşleme) modülü başlatılıyor...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Cihaz: {self.device}")

        # Takipçi (Tracker) durumları
        self.trackers = {}
        self.frame_count = 0

        # --- LightGlue Yükleme ---
        try:
            from lightglue import LightGlue, SuperPoint
            self.extractor = SuperPoint(max_num_keypoints=LIGHTGLUE_MAX_KEYPOINTS).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
            self.use_lightglue = True
            log.info("SuperPoint + LightGlue yüklendi.")
        except ImportError:
            log.warning("LightGlue bulunamadı, OpenCV SIFT fallback aktif.")
            self.sift = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self.use_lightglue = False

        # --- MobileSAM Yükleme ---
        self.use_sam = False
        if USE_SAM_REFINEMENT and os.path.exists(MOBILESAM_MODEL_PATH):
            try:
                from mobile_sam import sam_model_registry, SamPredictor
                sam = sam_model_registry["vit_t"](checkpoint=MOBILESAM_MODEL_PATH).to(self.device)
                sam.eval()
                self.sam_predictor = SamPredictor(sam)
                self.use_sam = True
                log.info("MobileSAM yüklendi. Kutu iyileştirme (refinement) aktif.")
            except ImportError:
                log.warning("MobileSAM kütüphanesi bulunamadı. Kutu iyileştirme devre dışı.")
            except Exception as e:
                log.warning(f"MobileSAM yüklenirken hata: {e}. Kutu iyileştirme devre dışı.")
        elif USE_SAM_REFINEMENT:
            log.warning(f"MobileSAM ağırlık dosyası bulunamadı: {MOBILESAM_MODEL_PATH}")

        log.info("Görev 3 modülü hazır.")

    # =========================================================================
    # ANA ÇALIŞMA DÖNGÜSÜ
    # =========================================================================
    def run(self, frame, reference_objects=None, **kwargs):
        """
        Her frame için çağrılır.
        
        Args:
            frame: Ana görüntü (numpy array, BGR).
            reference_objects: [{"id": int, "image": numpy_array}, ...] formatında referans objeler.
        
        Returns:
            Eşleşen objelerin listesi (şartname JSON formatında).
        """
        matched_objects = []
        self.frame_count += 1

        if not reference_objects:
            return matched_objects

        try:
            for ref_obj in reference_objects:
                obj_id = ref_obj["id"]
                ref_img = ref_obj["image"]

                # 6 objenin 6'sı da aynı anda LightGlue'ya girmesin diye offset (gecikme) ekliyoruz
                try:
                    offset = int(obj_id) * 3
                except (ValueError, TypeError):
                    offset = sum(ord(c) for c in str(obj_id)) * 3

                needs_detection = (
                    obj_id not in self.trackers or
                    (self.frame_count + offset) % TRACKER_REDETECTION_INTERVAL == 0
                )

                if needs_detection:
                    bbox, match_points = self._detect(frame, ref_img)

                    if bbox is not None:
                        # MobileSAM ile kutuyu iyileştir
                        if self.use_sam and match_points is not None:
                            refined_bbox = self._refine_with_sam(frame, match_points, bbox)
                            if refined_bbox is not None:
                                bbox = refined_bbox

                        # Tracker başlat
                        self._init_tracker(obj_id, frame, bbox)

                        matched_objects.append(self._format_bbox(obj_id, bbox))
                        log.info(f"Obje {obj_id} tespit edildi ve takibe alındı. BBox: {bbox}")
                    else:
                        log.debug(f"Obje {obj_id} bu karede bulunamadı.")
                        self.trackers.pop(obj_id, None)
                else:
                    # Tracker ile takip et
                    tracked_bbox = self._track(obj_id, frame)
                    if tracked_bbox is not None:
                        matched_objects.append(self._format_bbox(obj_id, tracked_bbox))
                    else:
                        log.warning(f"Obje {obj_id} takibi kaybedildi, sonraki karede tekrar aranacak.")

        except Exception as e:
            log.error(f"Görev 3 çalışırken hata: {e}", exc_info=True)

        return matched_objects

    # =========================================================================
    # TESPİT (DETECTION) KATMANI
    # =========================================================================
    def _detect(self, frame, ref_img):
        """LightGlue veya SIFT ile eşleştirme yapar. bbox ve eşleşen noktaları döndürür."""
        if self.use_lightglue:
            return self._match_lightglue(frame, ref_img)
        else:
            bbox = self._match_sift(frame, ref_img)
            return bbox, None

    def _match_lightglue(self, frame, ref_img):
        """SuperPoint + LightGlue ile referans objeyi ana karede arar."""
        from lightglue.utils import rbd

        # Ön işleme: CLAHE ile kontrast iyileştirmesi
        ref_enhanced = enhance_image(ref_img)
        frame_enhanced = enhance_image(frame)

        img0 = self._to_tensor(ref_enhanced)
        img1 = self._to_tensor(frame_enhanced)

        with torch.inference_mode():
            feats0 = self.extractor.extract(img0)
            feats1 = self.extractor.extract(img1)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})

        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']

        if len(matches) < LIGHTGLUE_MIN_MATCHES:
            return None, None

        m_kpts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        m_kpts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()

        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 3.0)
        if H is None:
            return None, None

        # Geometrik doğrulama: Saçma homografileri reddet
        if not verify_homography(H, ref_img.shape):
            log.debug("Homografi geometrik doğrulamadan geçemedi, atlanıyor.")
            return None, None

        # Güven skoru hesapla
        confidence = compute_match_confidence(matches, m_kpts0, m_kpts1, mask)
        if confidence < 0.45:
            log.debug(f"Eşleşme güven skoru çok düşük: {confidence:.3f}, atlanıyor.")
            return None, None
        log.debug(f"Eşleşme güven skoru: {confidence:.3f}")

        # Referans görüntünün köşelerini ana kareye yansıt
        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        bbox = self._corners_to_bbox(transformed, frame.shape)
        if bbox is None:
            return None, None

        # Kutu boyut doğrulaması: Tespit edilen kutu mantıklı mı?
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox_area = bbox_w * bbox_h
        frame_h, frame_w = frame.shape[:2]
        frame_area = frame_w * frame_h
        ref_h, ref_w = ref_img.shape[:2]
        ref_area = ref_h * ref_w

        # 1) Kutu, karenin %40'ından büyükse yanlış tespit
        if bbox_area > frame_area * 0.4:
            log.debug(f"Tespit edilen kutu cok buyuk ({bbox_area/frame_area:.1%} kare alani), atlaniyor.")
            return None, None

        # 2) Kutu, referans objenin 5 katından büyükse yanlış tespit
        if bbox_area > ref_area * 5:
            log.debug(f"Tespit edilen kutu referansa gore cok buyuk ({bbox_area/ref_area:.1f}x), atlaniyor.")
            return None, None

        # 3) Kutu, referans objenin %15'inden küçükse yanlış tespit (çok küçük parça)
        if bbox_area < ref_area * 0.15:
            log.debug(f"Tespit edilen kutu cok kucuk ({bbox_area/ref_area:.2f}x referans), atlaniyor.")
            return None, None

        # 4) En-boy oranı kontrolü: Referans ile tespitin oranı çok farklıysa reddet
        ref_ratio = ref_w / ref_h
        bbox_ratio = bbox_w / bbox_h
        ratio_diff = max(ref_ratio, bbox_ratio) / max(min(ref_ratio, bbox_ratio), 0.01)
        if ratio_diff > 3.0:
            log.debug(f"En-boy orani uyumsuz (ref: {ref_ratio:.2f}, tespit: {bbox_ratio:.2f}), atlaniyor.")
            return None, None

        # Inlier eşleşme noktalarını SAM için sakla
        inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(len(m_kpts1), dtype=bool)
        inlier_points = m_kpts1[inlier_mask]

        return bbox, inlier_points

    def _match_sift(self, frame, ref_img):
        """OpenCV SIFT fallback."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img

        kp1, des1 = self.sift.detectAndCompute(gray_ref, None)
        kp2, des2 = self.sift.detectAndCompute(gray_frame, None)

        if des1 is None or des2 is None:
            return None

        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        if len(matches) < LIGHTGLUE_MIN_MATCHES:
            return None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None

        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(corners, H)

        return self._corners_to_bbox(transformed, frame.shape)

    # =========================================================================
    # SAM İYİLEŞTİRME (REFINEMENT) KATMANI
    # =========================================================================
    def _refine_with_sam(self, frame, match_points, fallback_bbox):
        """
        MobileSAM ile eşleşen noktalardan piksel düzeyinde maske çıkarır.
        Maskeden elde edilen kutu, LightGlue kutusundan çok daha hassastır.
        
        Strateji: Hem nokta hem de kutu promptu birlikte verilir.
        - Nokta promptları: "Obje buralarda" ipucu
        - Kutu promptu: "Obje kabaca bu alanda" ipucu
        Bu ikisinin birleşimi, tek başına noktadan çok daha isabetli maske çıkarır.
        """
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(frame_rgb)

            # En fazla 32 nokta ver (çok fazla nokta SAM'ı yanıltabilir)
            if len(match_points) > 32:
                indices = np.random.choice(len(match_points), 32, replace=False)
                input_points = match_points[indices]
            else:
                input_points = match_points

            input_labels = np.ones(len(input_points))  # Hepsi pozitif (objenin içi)

            # Kutu promptu: LightGlue'nun bulduğu kabataslak kutuyu SAM'a ver
            input_box = np.array([fallback_bbox[0], fallback_bbox[1], 
                                  fallback_bbox[2], fallback_bbox[3]])

            masks, scores, _ = self.sam_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                box=input_box,
                multimask_output=True,
            )

            # En yüksek skorlu maskeyi seç
            best_idx = np.argmax(scores)
            mask = masks[best_idx]

            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0:
                return None

            frame_h, frame_w = frame.shape[:2]
            tl_x = max(0, int(np.min(x_indices)))
            tl_y = max(0, int(np.min(y_indices)))
            br_x = min(frame_w, int(np.max(x_indices)))
            br_y = min(frame_h, int(np.max(y_indices)))

            if (br_x - tl_x) < 10 or (br_y - tl_y) < 10:
                return None

            log.debug(f"SAM refinement: {fallback_bbox} -> ({tl_x}, {tl_y}, {br_x}, {br_y})")
            return (tl_x, tl_y, br_x, br_y)

        except Exception as e:
            log.warning(f"SAM iyileştirme başarısız: {e}. Orijinal kutu kullanılıyor.")
            return None

    # =========================================================================
    # TAKİP (TRACKING) KATMANI
    # =========================================================================
    def _init_tracker(self, obj_id, frame, bbox):
        """Yeni bir CSRT tracker başlatır."""
        tracker = cv2.TrackerCSRT_create()
        x, y = int(bbox[0]), int(bbox[1])
        w, h = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
        tracker.init(frame, (x, y, w, h))
        self.trackers[obj_id] = tracker

    def _track(self, obj_id, frame):
        """Mevcut tracker'ı günceller ve bbox döndürür."""
        if obj_id not in self.trackers:
            return None

        success, bbox_tracker = self.trackers[obj_id].update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox_tracker]
            return (x, y, x + w, y + h)
        else:
            del self.trackers[obj_id]
            return None

    # =========================================================================
    # YARDIMCI FONKSİYONLAR
    # =========================================================================
    def _to_tensor(self, img):
        """Numpy BGR görüntüyü LightGlue'nun beklediği (C, H, W) RGB float32 tensöre çevirir."""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        return tensor.to(self.device)

    def _corners_to_bbox(self, corners, frame_shape):
        """Perspektif dönüşümünden gelen köşe noktalarını (tl_x, tl_y, br_x, br_y) bbox'a çevirir."""
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]

        frame_h, frame_w = frame_shape[:2]
        tl_x = max(0, float(min(x_coords)))
        tl_y = max(0, float(min(y_coords)))
        br_x = min(frame_w, float(max(x_coords)))
        br_y = min(frame_h, float(max(y_coords)))

        if (br_x - tl_x) < 10 or (br_y - tl_y) < 10:
            return None

        return (tl_x, tl_y, br_x, br_y)

    def _format_bbox(self, obj_id, bbox):
        """Kutuyu şartname JSON formatına dönüştürür."""
        return format_undefined_object_output(
            obj_id=obj_id,
            top_left_x=bbox[0],
            top_left_y=bbox[1],
            bottom_right_x=bbox[2],
            bottom_right_y=bbox[3]
        )
