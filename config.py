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

# Görev 3 (Görüntü Eşleme) Ayarları — DINOv2 + LoFTR Hibrit Mimari
DINOV2_MODEL_NAME = "dinov2_vits14"        # En küçük ve hızlı DINOv2 modeli
DINOV2_SIMILARITY_THRESHOLD = 0.25         # Patch benzerlik eşiği (düşükse daha çok aday bulunur)
DINOV2_INPUT_SIZE = 518                    # 14'ün katı olmalı (37 patch = 518px)
DINOV2_TOP_K_CANDIDATES = 3               # Her referans için en fazla kaç aday bölge

LOFTR_PRETRAINED = "outdoor"               # Drone videoları için outdoor modeli
LOFTR_MIN_MATCHES = 8                      # Minimum güvenilir eşleşme sayısı
LOFTR_CONFIDENCE_THRESHOLD = 0.3           # LoFTR eşleşme güven eşiği
LOFTR_MAX_IMAGE_SIZE = 480                 # LoFTR girdi boyutu (hız için sınırla)

MOBILESAM_MODEL_PATH = "models/mobile_sam.pt"  # MobileSAM ağırlık dosyası
USE_SAM_REFINEMENT = False                 # SAM şimdilik devre dışı (hız için)

TRACKER_REDETECTION_INTERVAL = 15          # Kaç karede bir tekrar tespit yapılacak
