import cv2
import numpy as np
from config import (
    CALIBRATION_FILE_PATH,
    DEFAULT_CAMERA_TYPE,
    ORB_MAX_FEATURES,
    LOWE_RATIO_THRESHOLD,
    BLUR_LAPLACIAN_THRESHOLD,
    DEFAULT_FLIGHT_ALTITUDE,
)
from utils.calibration_parser import parse_calibration_file
from utils.helper import format_translation_output
from utils.logger import log


class PositionEstimator:
    """
    Görev 2: Görsel Odometri (Visual Odometry) ile GPS-Bağımsız Konum Kestirimi.

    - Kamera kalibrasyon matrisi (K) ve distorsiyon katsayıları ile piksel->metre dönüşümü
    - ORB öznitelik çıkarımı + Hamming BFMatcher (Lowe's Ratio Test) + RANSAC afin kestirim
    - Laplacian varyansı ile hareket bulanıklığı (motion blur) kontrolü
    - 6-durumlu (x, y, z, vx, vy, vz) Kalman Filtresi ile gürültü ve sıçrama bastırma
    - GPS sağlık durumu (health_status == 1) geldiğinde birikimli sürüklenme (drift) sıfırlama
    """

    FALLBACK_INTRINSICS = {
        "K": np.array(
            [[1389.7, 0.0, 954.0], [0.0, 1387.1, 558.9], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        "dist_coeffs": np.zeros(5, dtype=np.float32),
    }

    def __init__(self, camera_type=DEFAULT_CAMERA_TYPE):
        log.info("Görev 2 (Pozisyon Kestirimi) modülü başlatılıyor...")
        self.camera_type = camera_type
        self.calibrations = parse_calibration_file(CALIBRATION_FILE_PATH)
        if self.calibrations is None:
            self.calibrations = parse_calibration_file()

        self.orb = cv2.ORB_create(nfeatures=ORB_MAX_FEATURES)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray = None
        self.prev_kp = None
        self.prev_desc = None
        self.position = np.zeros(3, dtype=np.float64)
        self.kalman = self._create_kalman_filter()

        log.info(
            f"Görev 2 hazır (Kamera profili: {self.camera_type}, "
            f"ORB özellik limiti: {ORB_MAX_FEATURES})."
        )

    @staticmethod
    def _create_kalman_filter():
        """Sabit hız modelli 3B konum + hız Kalman Filtresi oluşturur."""
        kf = cv2.KalmanFilter(6, 3)
        kf.transitionMatrix = np.array(
            [
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 0.85, 0, 0],
                [0, 0, 0, 0, 0.85, 0],
                [0, 0, 0, 0, 0, 0.85],
            ],
            dtype=np.float32,
        )
        kf.measurementMatrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 5e-2
        kf.errorCovPost = np.eye(6, dtype=np.float32) * 0.1
        kf.statePost = np.zeros((6, 1), dtype=np.float32)
        return kf

    def _get_intrinsics(self, camera_type=None):
        """Seçili kamera profiline ait K matrisi ve distorsiyon vektörünü döndürür."""
        cam = camera_type or self.camera_type
        if self.calibrations and cam in self.calibrations:
            return self.calibrations[cam]
        return self.FALLBACK_INTRINSICS

    @staticmethod
    def _is_blurry(gray):
        """Laplacian varyansı ile karede aşırı hareket bulanıklığı olup olmadığını ölçer."""
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return lap_var < BLUR_LAPLACIAN_THRESHOLD, lap_var

    def _reset_with_gps(self, ref_pos):
        """GPS sağlıklı olduğunda konumu ve Kalman durumunu referans pozisyona eşitler."""
        self.position = np.array(ref_pos, dtype=np.float64)
        self.kalman.statePost = np.array(
            [[ref_pos[0]], [ref_pos[1]], [ref_pos[2]], [0.0], [0.0], [0.0]],
            dtype=np.float32,
        )
        self.kalman.statePre = self.kalman.statePost.copy()

    def _predict_only(self):
        """Ölçüm yapılamayan/bulanık karelerde Kalman tahminiyle devam eder."""
        pred = self.kalman.predict()
        self.position = pred[:3, 0].astype(np.float64)

    def _estimate_visual_delta(self, gray, intrinsics, altitude):
        """
        Ardışık kareler arasında ORB özniteliklerini eşleştirerek
        kamera merkezinin metrik yer değiştirmesini (dx, dy, dz) hesaplar.
        """
        kp, desc = self.orb.detectAndCompute(gray, None)
        if (
            self.prev_kp is None
            or self.prev_desc is None
            or desc is None
            or len(kp) < 8
            or len(self.prev_kp) < 8
        ):
            return None, kp, desc

        knn_matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < LOWE_RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        if len(good_matches) < 8:
            return None, kp, desc

        pts_prev = np.float32(
            [self.prev_kp[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        pts_curr = np.float32(
            [kp[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        K = intrinsics["K"]
        dist = intrinsics["dist_coeffs"]
        pts_prev_u = cv2.undistortPoints(pts_prev, K, dist, P=K)
        pts_curr_u = cv2.undistortPoints(pts_curr, K, dist, P=K)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_prev_u, pts_curr_u, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None or inliers is None or int(np.sum(inliers)) < 6:
            return None, kp, desc

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        # Dönme kaynaklı kaldıraç kolunu elemek için optik merkezdeki yer değiştirmeyi ölç
        center_moved = M @ np.array([cx, cy, 1.0], dtype=np.float64)
        dx_px = float(center_moved[0] - cx)
        dy_px = float(center_moved[1] - cy)
        scale = float(np.hypot(M[0, 0], M[1, 0]))

        dx_m = -(dx_px / max(fx, 1e-6)) * altitude
        dy_m = -(dy_px / max(fy, 1e-6)) * altitude
        if 0.5 <= scale <= 2.0:
            dz_m = altitude * (1.0 - 1.0 / scale)
        else:
            dz_m = 0.0

        return np.array([dx_m, dy_m, dz_m], dtype=np.float64), kp, desc

    def run(self, current_frame, **kwargs):
        """
        Ardışık karelerden X, Y, Z eksenlerindeki konum kestirimini hesaplar.

        Args:
            current_frame: OpenCV BGR (veya gri) görüntü matrisi.
            **kwargs:
                health_status (int): GPS sağlık durumu (1: sağlıklı, 0: kesik).
                translation_x / ref_x (float): GPS referans X konumu.
                translation_y / ref_y (float): GPS referans Y konumu.
                translation_z / ref_z (float): GPS referans Z konumu.
                altitude (float): Metre cinsinden uçuş yüksekliği (H).
                camera_type (str): "Termal", "RGB 4K" veya "RGB 1080p".

        Returns:
            TEAM_INTERFACE formatına uygun tek elemanlı sözlük listesi.
        """
        translations = []
        if current_frame is None or getattr(current_frame, "size", 0) == 0:
            return [format_translation_output(*self.position)]

        try:
            gray = (
                cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                if len(current_frame.shape) == 3
                else current_frame.copy()
            )
            intrinsics = self._get_intrinsics(kwargs.get("camera_type"))
            altitude = float(kwargs.get("altitude", DEFAULT_FLIGHT_ALTITUDE))
            health_status = int(kwargs.get("health_status", 0))

            ref_x = kwargs.get("ref_x", kwargs.get("translation_x"))
            ref_y = kwargs.get("ref_y", kwargs.get("translation_y"))
            ref_z = kwargs.get("ref_z", kwargs.get("translation_z"))

            if health_status == 1 and None not in (ref_x, ref_y, ref_z):
                self._reset_with_gps((float(ref_x), float(ref_y), float(ref_z)))
                is_blurry, _ = self._is_blurry(gray)
                if not is_blurry:
                    kp, desc = self.orb.detectAndCompute(gray, None)
                    self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
                translations.append(format_translation_output(*self.position))
                return translations

            is_blurry, lap_var = self._is_blurry(gray)
            if self.prev_gray is None or self.prev_gray.shape != gray.shape:
                if not is_blurry:
                    kp, desc = self.orb.detectAndCompute(gray, None)
                    self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc
                else:
                    self.prev_gray = gray
                translations.append(format_translation_output(*self.position))
                return translations

            if is_blurry:
                log.debug(
                    f"Görev 2: Bulanık kare (Laplacian={lap_var:.1f}), Kalman tahmini kullanılıyor."
                )
                self._predict_only()
            else:
                delta, kp, desc = self._estimate_visual_delta(gray, intrinsics, altitude)
                if delta is not None:
                    raw_pos = self.position + delta
                    self.kalman.predict()
                    corrected = self.kalman.correct(
                        raw_pos.reshape(3, 1).astype(np.float32)
                    )
                    self.position = corrected[:3, 0].astype(np.float64)
                else:
                    self._predict_only()

                self.prev_gray, self.prev_kp, self.prev_desc = gray, kp, desc

            translations.append(format_translation_output(*self.position))
            log.debug(
                f"Görev 2 konum kestirimi: x={self.position[0]:.3f}, "
                f"y={self.position[1]:.3f}, z={self.position[2]:.3f}"
            )
        except Exception as e:
            log.error(f"Görev 2 çalışırken hata: {e}", exc_info=True)
            translations.append(format_translation_output(*self.position))

        return translations
