from config import TASK1_MODEL_PATH, CONFIDENCE_THRESHOLD
from utils.helper import format_detection_output
from utils.logger import log

class ObjectDetector:
    def __init__(self):
        log.info(f"Görev 1 (Nesne Tespiti) modeli yükleniyor: {TASK1_MODEL_PATH}")
        # self.model = load_model(TASK1_MODEL_PATH)

    def run(self, frame, **kwargs):
        """
        Frame üzerinde nesne tespiti yapar.
        Döndürdüğü değer: formata uygun tespit edilmiş objelerin listesi.
        """
        detected_objects = []
        try:
            # Örnek dummy tespit: Taşıt(0), Hareketli(1), İniş Durumu Uygun Değil(0)
            dummy_obj = format_detection_output(
                cls=0,
                landing_status=0,
                motion_status=1,
                top_left_x=100.5,
                top_left_y=200.0,
                bottom_right_x=300.2,
                bottom_right_y=400.1
            )
            detected_objects.append(dummy_obj)
            log.debug(f"Görev 1 tespit edilen nesne sayısı: {len(detected_objects)}")
        except Exception as e:
            log.error(f"Görev 1 çalışırken hata oluştu: {e}")
            
        return detected_objects
