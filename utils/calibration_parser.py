import os
import numpy as np
from utils.logger import log

DEFAULT_CALIBRATION_PATHS = (
    os.path.join("data", "Kamera_Kalibrasyon_Parametreleri_2026.txt"),
    os.path.join("veriseti_2026", "THYZ_2026_Ornek_Veri_Seti", "Kamera_Kalibrasyon_Parametreleri_2026.txt"),
)


def resolve_calibration_path(file_path=None):
    """Varsayılan kalibrasyon dosyası yollarını sırayla kontrol eder."""
    if file_path is not None:
        return file_path

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for candidate in DEFAULT_CALIBRATION_PATHS:
        if os.path.exists(candidate):
            return candidate
        abs_candidate = os.path.join(base_dir, candidate)
        if os.path.exists(abs_candidate):
            return abs_candidate
    return DEFAULT_CALIBRATION_PATHS[0]


def parse_calibration_file(file_path=None):
    """
    Kamera_Kalibrasyon_Parametreleri_2026.txt dosyasını parse eder.
    
    Döndürdüğü yapı:
    {
        "Termal": {
            "K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
            "dist_coeffs": np.array([k1, k2, p1, p2, 0]) # k3=0 genelde
        },
        "RGB 4K": { ... },
        "RGB 1080p": { ... }
    }
    """
    resolved_path = resolve_calibration_path(file_path)
    if not os.path.exists(resolved_path):
        log.error(f"Kalibrasyon dosyası bulunamadı: {resolved_path}")
        return None

    cameras = {}
    current_cam = None

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Kamera adını yakala (örn: "Termal Camera Intrinsics")
                if "Camera" in line and "Intrinsics" in line:
                    if "Termal" in line:
                        current_cam = "Termal"
                    elif "4K" in line:
                        current_cam = "RGB 4K"
                    elif "1080p" in line:
                        current_cam = "RGB 1080p"
                    else:
                        current_cam = line
                        
                    cameras[current_cam] = {
                        "fx": 0.0, "fy": 0.0,
                        "cx": 0.0, "cy": 0.0,
                        "k1": 0.0, "k2": 0.0,
                        "p1": 0.0, "p2": 0.0
                    }
                    continue
                
                if current_cam is None:
                    continue

                # Parametreleri parse et
                if "FocalLength:" in line:
                    # FocalLength: [731.7965 732.0172]
                    parts = line.split("[")[1].split("]")[0].split()
                    cameras[current_cam]["fx"] = float(parts[0])
                    cameras[current_cam]["fy"] = float(parts[1])
                    
                elif "PrincipalPoint:" in line:
                    parts = line.split("[")[1].split("]")[0].split()
                    cameras[current_cam]["cx"] = float(parts[0])
                    cameras[current_cam]["cy"] = float(parts[1])
                    
                elif "RadialDistortion:" in line:
                    parts = line.split("[")[1].split("]")[0].split()
                    cameras[current_cam]["k1"] = float(parts[0])
                    cameras[current_cam]["k2"] = float(parts[1])
                    
                elif "TangentialDistortion:" in line:
                    parts = line.split("[")[1].split("]")[0].split()
                    cameras[current_cam]["p1"] = float(parts[0])
                    cameras[current_cam]["p2"] = float(parts[1])

        # Değerleri OpenCV'nin beklediği Matrix formatına çevir
        result = {}
        for cam, params in cameras.items():
            K = np.array([
                [params["fx"], 0, params["cx"]],
                [0, params["fy"], params["cy"]],
                [0, 0, 1]
            ], dtype=np.float32)
            
            dist_coeffs = np.array([
                params["k1"], params["k2"], params["p1"], params["p2"], 0.0
            ], dtype=np.float32)
            
            result[cam] = {"K": K, "dist_coeffs": dist_coeffs}
            
        log.info(f"Kalibrasyon dosyası başarıyla okundu: {list(result.keys())}")
        return result

    except Exception as e:
        log.error(f"Kalibrasyon parse edilirken hata oluştu: {e}")
        return None

if __name__ == "__main__":
    # Test
    res = parse_calibration_file()
    if res:
        print("1080p K Matrisi:")
        print(res.get("RGB 1080p", {}).get("K"))
