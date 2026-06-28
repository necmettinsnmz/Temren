# Sunucu Ayarları
SERVER_URL = "http://127.0.0.25:5000"
USER_URL = "http://localhost/users/1/"  # Kendi takımınızın ID'sine göre güncellenecektir.
API_ENDPOINT_RESULTS = f"{SERVER_URL}/api/submit"  # Örnek API ucu, yarışma günü kesinleşecek.

# Görüntü Çözünürlüğü ve Parametreleri
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Görev 1 (Nesne Tespiti) Ayarları
# Örnek: YOLOv8 vb. ağırlık yolları eklenebilir.
TASK1_MODEL_PATH = "models/yolov8_custom.pt"
CONFIDENCE_THRESHOLD = 0.5

# Görev 2 (Pozisyon Tespiti) Ayarları
# Optik akış veya diğer parametreler
FEATURE_PARAMS = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Görev 3 (Görüntü Eşleme) Ayarları
LIGHTGLUE_MAX_KEYPOINTS = 2048          # SuperPoint'in çıkaracağı max özellik noktası
LIGHTGLUE_MIN_MATCHES = 4              # Homografi hesabı için minimum eşleşme sayısı
MOBILESAM_MODEL_PATH = "models/mobile_sam.pt"  # MobileSAM ağırlık dosyası
USE_SAM_REFINEMENT = True              # MobileSAM ile kutu iyileştirme açık/kapalı
MATCHING_THRESHOLD = 0.7
TRACKER_REDETECTION_INTERVAL = 15      # Kaç karede bir LightGlue ile tekrar doğrulama yapılacak
