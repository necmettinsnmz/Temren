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
# SIFT, ORB vb. eşleme eşik değerleri
MATCHING_THRESHOLD = 0.7
