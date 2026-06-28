import numpy as np
from config import FEATURE_PARAMS
from utils.helper import format_translation_output
from utils.logger import log

class PositionEstimator:
    def __init__(self):
        log.info("Görev 2 (Pozisyon Kestirimi) modülü başlatıldı.")
        self.previous_frame = None
        self.previous_features = None

    def run(self, current_frame, **kwargs):
        """
        Optik akış veya SLAM ile ardışık framelerden X, Y, Z yer değiştirmesini hesaplar.
        """
        translations = []
        try:
            # Şimdilik dummy bir yer değiştirme ekleyelim.
            dummy_trans = format_translation_output(tx=0.5, ty=-0.2, tz=0.0)
            translations.append(dummy_trans)
            
            # Frame'i bir sonraki iterasyon için kaydet
            self.previous_frame = current_frame.copy() if current_frame is not None else None
            log.debug("Görev 2 pozisyon hesaplaması tamamlandı.")
        except Exception as e:
            log.error(f"Görev 2 çalışırken hata: {e}")
            
        return translations
