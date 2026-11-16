import os
import cv2
import numpy as np
from config import (
    TASK1_MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    IOU_NMS_THRESHOLD,
    USE_SAHI_SLICING,
    SAHI_SLICE_SIZE,
    SAHI_OVERLAP_RATIO,
    MOTION_FLOW_THRESHOLD,
    LANDING_IOU_THRESHOLD,
    FEATURE_PARAMS,
)
from utils.helper import format_detection_output
from utils.logger import log


class ObjectDetector:
    """
    Görev 1: Nesne Tespiti, Hareket Analizi ve İniş Alanı Uygunluk Değerlendirmesi.

    Sınıflar:
        0: Taşıt
        1: İnsan
        2: UAP (Uçan Araba Park Alanı)
        3: UAİ (Uçan Ambulans İniş Alanı)
    """

    VEHICLE_CLS = 0
    HUMAN_CLS = 1
    LANDING_CLASSES = (2, 3)
    OBSTACLE_CLASSES = (0, 1)

    def __init__(self):
        log.info(f"Görev 1 (Nesne Tespiti) başlatılıyor: {TASK1_MODEL_PATH}")
        self.model = None
        self.prev_gray = None

        if os.path.exists(TASK1_MODEL_PATH):
            try:
                from ultralytics import YOLO

                self.model = YOLO(TASK1_MODEL_PATH)
                log.info("Görev 1 YOLOv8 modeli başarıyla yüklendi.")
            except ImportError:
                log.warning(
                    "ultralytics paketi kurulu değil; Görev 1 yalnızca harici/test "
                    "tespitleriyle (fallback) çalışacak."
                )
            except Exception as e:
                log.error(f"Görev 1 modeli yüklenirken hata oluştu: {e}")
        else:
            log.warning(
                f"Model ağırlığı bulunamadı ({TASK1_MODEL_PATH}). "
                "Görev 1 fallback modunda çalışacak."
            )

    def run(self, frame, **kwargs):
        """
        Frame üzerinde nesne tespiti, hareket durumu ve iniş alanı analizi yapar.

        Args:
            frame: OpenCV BGR (veya gri) görüntü matrisi.
            **kwargs:
                use_sahi (bool): SAHI dilimli çıkarım bayrağı.
                raw_detections / detections (list): Test veya harici model çıktısı enjeksiyonu.

        Returns:
            TEAM_INTERFACE formatına uygun sözlük listesi.
        """
        detected_objects = []
        if frame is None or getattr(frame, "size", 0) == 0:
            return detected_objects

        try:
            gray = (
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if len(frame.shape) == 3
                else frame.copy()
            )

            raw_detections = self._get_raw_detections(frame, **kwargs)
            ego_homography = self._estimate_ego_homography(gray)

            obstacles = [
                det[:5] for det in raw_detections if det[0] in self.OBSTACLE_CLASSES
            ]

            for cls_id, x1, y1, x2, y2, _conf in raw_detections:
                if cls_id == self.VEHICLE_CLS:
                    motion_status = self._estimate_motion_status(
                        gray, (x1, y1, x2, y2), ego_homography
                    )
                else:
                    motion_status = -1

                if cls_id in self.LANDING_CLASSES:
                    landing_status = self._evaluate_landing_status(
                        (x1, y1, x2, y2), obstacles
                    )
                else:
                    landing_status = -1

                detected_objects.append(
                    format_detection_output(
                        cls=cls_id,
                        landing_status=landing_status,
                        motion_status=motion_status,
                        top_left_x=x1,
                        top_left_y=y1,
                        bottom_right_x=x2,
                        bottom_right_y=y2,
                    )
                )

            self.prev_gray = gray
            log.debug(f"Görev 1 tespit edilen nesne sayısı: {len(detected_objects)}")
        except Exception as e:
            log.error(f"Görev 1 çalışırken hata oluştu: {e}", exc_info=True)

        return detected_objects

    def _get_raw_detections(self, frame, **kwargs):
        """Harici/test enjeksiyonu varsa onu okur, yoksa YOLOv8 çıkarımı yapar."""
        injected = kwargs.get("raw_detections", kwargs.get("detections"))
        if injected is not None:
            normalized = []
            for item in injected:
                if isinstance(item, dict):
                    normalized.append((
                        int(item["cls"]),
                        float(item["top_left_x"]),
                        float(item["top_left_y"]),
                        float(item["bottom_right_x"]),
                        float(item["bottom_right_y"]),
                        float(item.get("conf", 1.0)),
                    ))
                else:
                    cls_id, x1, y1, x2, y2 = item[:5]
                    conf = float(item[5]) if len(item) > 5 else 1.0
                    normalized.append((int(cls_id), float(x1), float(y1), float(x2), float(y2), conf))
            return normalized

        if self.model is None:
            return []

        use_sahi = kwargs.get("use_sahi", USE_SAHI_SLICING)
        if use_sahi:
            return self._infer_sliced(frame)
        return self._infer_full(frame)

    def _infer_full(self, frame):
        """Tam kare üzerinde tek geçişli YOLOv8 çıkarımı."""
        results = self.model(
            frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_NMS_THRESHOLD, verbose=False
        )
        return self._extract_yolo_boxes(results)

    def _infer_sliced(self, frame):
        """
        SAHI (Slicing Aided Hyper Inference) yaklaşımıyla yüksek çözünürlüklü
        görüntüyü örtüşen alt yamalara bölerek küçük nesne tespitini güçlendirir.
        """
        h, w = frame.shape[:2]
        slice_size = SAHI_SLICE_SIZE
        if w <= slice_size and h <= slice_size:
            return self._infer_full(frame)

        stride = max(1, int(slice_size * (1.0 - SAHI_OVERLAP_RATIO)))
        xs = list(range(0, max(1, w - slice_size + 1), stride))
        ys = list(range(0, max(1, h - slice_size + 1), stride))
        if xs[-1] + slice_size < w:
            xs.append(max(0, w - slice_size))
        if ys[-1] + slice_size < h:
            ys.append(max(0, h - slice_size))

        all_detections = []
        for y0 in ys:
            for x0 in xs:
                x1_crop = min(w, x0 + slice_size)
                y1_crop = min(h, y0 + slice_size)
                patch = frame[y0:y1_crop, x0:x1_crop]
                results = self.model(
                    patch, conf=CONFIDENCE_THRESHOLD, iou=IOU_NMS_THRESHOLD, verbose=False
                )
                for cls_id, bx1, by1, bx2, by2, conf in self._extract_yolo_boxes(results):
                    all_detections.append(
                        (cls_id, bx1 + x0, by1 + y0, bx2 + x0, by2 + y0, conf)
                    )

        return self._apply_nms(all_detections)

    @staticmethod
    def _extract_yolo_boxes(results):
        """Ultralytics sonuç nesnesini (cls, x1, y1, x2, y2, conf) listesine çevirir."""
        detections = []
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append((cls_id, float(x1), float(y1), float(x2), float(y2), conf))
        return detections

    @staticmethod
    def _apply_nms(detections):
        """Sınıf bazlı Non-Maximum Suppression uygular."""
        if not detections:
            return []

        merged = []
        classes = {det[0] for det in detections}
        for cls_id in classes:
            cls_dets = [det for det in detections if det[0] == cls_id]
            boxes_xywh = [
                [det[1], det[2], max(1.0, det[3] - det[1]), max(1.0, det[4] - det[2])]
                for det in cls_dets
            ]
            scores = [det[5] for det in cls_dets]
            indices = cv2.dnn.NMSBoxes(
                boxes_xywh, scores, CONFIDENCE_THRESHOLD, IOU_NMS_THRESHOLD
            )
            if len(indices) == 0:
                continue
            for idx in np.array(indices).flatten():
                merged.append(cls_dets[int(idx)])
        return merged

    def _estimate_ego_homography(self, gray):
        """
        Ardışık kareler arasında arka plan kamera hareketini (ego-motion)
        Lucas-Kanade optik akışı ve RANSAC homografi ile kestirir.
        """
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return None

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **FEATURE_PARAMS)
        if prev_pts is None or len(prev_pts) < 8:
            return None

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None
        )
        if curr_pts is None or status is None:
            return None

        valid = status.ravel() == 1
        good_prev = prev_pts[valid].reshape(-1, 2)
        good_curr = curr_pts[valid].reshape(-1, 2)
        if len(good_prev) < 4:
            return None

        H, _ = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 3.0)
        return H

    def _estimate_motion_status(self, gray, bbox, ego_homography):
        """
        Taşıt kutusu içindeki artık optik akışı ego-motion'dan arındırarak
        taşıtın hareketli (1) veya hareketsiz (0) olduğunu belirler.
        """
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            return 0

        h, w = gray.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if (x2 - x1) < 8 or (y2 - y1) < 8:
            return 0

        roi_curr = gray[y1:y2, x1:x2]
        roi_pts = cv2.goodFeaturesToTrack(
            roi_curr, maxCorners=30, qualityLevel=0.01, minDistance=4, blockSize=5
        )
        if roi_pts is None or len(roi_pts) == 0:
            return 0

        curr_pts = roi_pts.copy()
        curr_pts[:, 0, 0] += x1
        curr_pts[:, 0, 1] += y1

        # Mevcut kutu içindeki noktaları bir önceki kareye geri takip et
        prev_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.prev_gray, curr_pts, None
        )
        if prev_pts is None or status is None:
            return 0

        valid = status.ravel() == 1
        if not np.any(valid):
            return 0

        pts_curr_valid = curr_pts[valid].reshape(-1, 2)
        pts_prev_valid = prev_pts[valid].reshape(-1, 2)

        if ego_homography is not None:
            expected_pts = cv2.perspectiveTransform(
                pts_prev_valid.reshape(-1, 1, 2).astype(np.float32), ego_homography
            ).reshape(-1, 2)
            residual_disp = np.linalg.norm(pts_curr_valid - expected_pts, axis=1)
        else:
            residual_disp = np.linalg.norm(pts_curr_valid - pts_prev_valid, axis=1)

        median_disp = float(np.median(residual_disp))
        return 1 if median_disp > MOTION_FLOW_THRESHOLD else 0

    @staticmethod
    def _evaluate_landing_status(landing_bbox, obstacles):
        """
        UAP/UAİ iniş alanının diğer nesnelerle (taşıt/insan) çakışmasını kontrol eder.
        Çakışma eşiği aşılırsa inişe uygun değil (0), temizse uygun (1) döner.
        """
        lx1, ly1, lx2, ly2 = landing_bbox
        landing_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        if landing_area <= 0.0:
            return 0

        for _obs_cls, ox1, oy1, ox2, oy2 in obstacles:
            ix1 = max(lx1, ox1)
            iy1 = max(ly1, oy1)
            ix2 = min(lx2, ox2)
            iy2 = min(ly2, oy2)

            inter_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter_area <= 0.0:
                continue

            obs_area = max(0.0, ox2 - ox1) * max(0.0, oy2 - oy1)
            union_area = landing_area + obs_area - inter_area
            iou = inter_area / union_area if union_area > 0.0 else 0.0
            containment = inter_area / obs_area if obs_area > 0.0 else 0.0

            if iou > LANDING_IOU_THRESHOLD or containment > LANDING_IOU_THRESHOLD:
                return 0

        return 1
