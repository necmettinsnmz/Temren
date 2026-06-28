import cv2
import numpy as np
import torch
from utils.helper import format_undefined_object_output
from utils.logger import log

class ImageMatcher:
    def __init__(self):
        log.info("Görev 3 (Görüntü Eşleme ve Takip) modülü başlatıldı.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log.info(f"Cihaz: {self.device}")
        
        # Takipçi (Tracker) sözlüğü: Her objenin ID'sine karşılık gelen aktif tracker'ı tutar.
        self.trackers = {}
        self.frame_count = 0
        self.detection_interval = 15  # Her 15 karede bir LightGlue ile doğrulama (Re-detection)
        
        try:
            from lightglue import LightGlue, SuperPoint
            self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='superpoint').eval().to(self.device)
            self.use_lightglue = True
            log.info("Orijinal SuperPoint + LightGlue modeli başarıyla yüklendi.")
        except ImportError:
            log.warning("LightGlue bulunamadı, OpenCV SIFT yöntemine düşülüyor.")
            self.sift = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            self.use_lightglue = False

    def run(self, frame, reference_objects=None, **kwargs):
        """
        frame: Ana görüntü (Numpy array)
        reference_objects: Aranacak referans objelerin listesi. Örn: [{"id": 1, "image": ref_img_numpy}]
        """
        matched_objects = []
        self.frame_count += 1

        if not reference_objects:
            log.debug("Referans obje listesi boş, eşleştirme atlanıyor.")
            return matched_objects
            
        try:
            for ref_obj in reference_objects:
                obj_id = ref_obj["id"]
                ref_img = ref_obj["image"]
                
                # Eğer objeyi şu an takip etmiyorsak VEYA doğrulama (detection_interval) zamanı geldiyse
                if obj_id not in self.trackers or self.frame_count % self.detection_interval == 0:
                    log.debug(f"[{obj_id}] için LightGlue tespit algoritması çalıştırılıyor...")
                    if self.use_lightglue:
                        bbox = self._match_lightglue(frame, ref_img)
                    else:
                        bbox = self._match_sift(frame, ref_img)
                    
                    if bbox is not None:
                        # Yeni Tracker başlat (CSRT daha yavaştır ama çok kesindir. KCF hızlıdır ama çabuk kaybeder)
                        # Yarışma formatında hassasiyet önemli olduğu için CSRT kullanıyoruz.
                        tracker = cv2.TrackerCSRT_create()
                        x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
                        tracker.init(frame, (x, y, w, h))
                        self.trackers[obj_id] = tracker
                        
                        matched_obj = format_undefined_object_output(
                            obj_id=obj_id,
                            top_left_x=bbox[0],
                            top_left_y=bbox[1],
                            bottom_right_x=bbox[2],
                            bottom_right_y=bbox[3]
                        )
                        matched_objects.append(matched_obj)
                        log.info(f"Referans Obje {obj_id} LightGlue ile bulundu ve takibe alındı.")
                    else:
                        log.debug(f"Referans Obje {obj_id} bulunamadı.")
                        if obj_id in self.trackers:
                            del self.trackers[obj_id] # Bulunamadıysa takibi de bırak
                else:
                    # Obje zaten takip ediliyorsa, ağır olan LightGlue yerine süper hızlı CSRT tracker'ı güncelle
                    success, bbox_tracker = self.trackers[obj_id].update(frame)
                    
                    if success:
                        x, y, w, h = [int(v) for v in bbox_tracker]
                        matched_obj = format_undefined_object_output(
                            obj_id=obj_id,
                            top_left_x=x,
                            top_left_y=y,
                            bottom_right_x=x+w,
                            bottom_right_y=y+h
                        )
                        matched_objects.append(matched_obj)
                        log.debug(f"Referans Obje {obj_id} Tracker ile takip ediliyor.")
                    else:
                        log.warning(f"Referans Obje {obj_id} takibi kaybedildi. Bir sonraki karede LightGlue ile tekrar aranacak.")
                        del self.trackers[obj_id]
                        
        except Exception as e:
            log.error(f"Görev 3 çalışırken hata: {e}")
            
        return matched_objects

    def _numpy_image_to_tensor(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        return tensor

    def _match_lightglue(self, frame, ref_img):
        from lightglue.utils import rbd
        
        img0 = self._numpy_image_to_tensor(ref_img).to(self.device)
        img1 = self._numpy_image_to_tensor(frame).to(self.device)
        
        with torch.inference_mode():
            feats0 = self.extractor.extract(img0)
            feats1 = self.extractor.extract(img1)
            matches01 = self.matcher({"image0": feats0, "image1": feats1})
            
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01['matches']
        
        if len(matches) < 4:
            return None
            
        m_kpts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
        m_kpts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
        
        H, mask = cv2.findHomography(m_kpts0, m_kpts1, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
            
        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        frame_h, frame_w = frame.shape[:2]
        tl_x = max(0, min(x_coords))
        tl_y = max(0, min(y_coords))
        br_x = min(frame_w, max(x_coords))
        br_y = min(frame_h, max(y_coords))
        
        if (br_x - tl_x) < 10 or (br_y - tl_y) < 10:
            return None
            
        return (tl_x, tl_y, br_x, br_y)

    def _match_sift(self, frame, ref_img):
        # SIFT Fallback code unchanged
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY) if len(ref_img.shape) == 3 else ref_img
        
        kp1, des1 = self.sift.detectAndCompute(gray_ref, None)
        kp2, des2 = self.sift.detectAndCompute(gray_frame, None)
        
        if des1 is None or des2 is None:
            return None
            
        matches = self.bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        if len(good_matches) < 4:
            return None
            
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            return None
            
        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        x_coords = transformed_corners[:, 0, 0]
        y_coords = transformed_corners[:, 0, 1]
        
        frame_h, frame_w = frame.shape[:2]
        tl_x = max(0, min(x_coords))
        tl_y = max(0, min(y_coords))
        br_x = min(frame_w, max(x_coords))
        br_y = min(frame_h, max(y_coords))
        
        if (br_x - tl_x) < 10 or (br_y - tl_y) < 10:
            return None
            
        return (tl_x, tl_y, br_x, br_y)
