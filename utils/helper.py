import cv2
import numpy as np

def load_image_from_url(image_url):
    """
    Sunucudan veya yerel diskten frame yükleme fonksiyonu.
    """
    # Gerçek uygulamada sunucudan indirilecek veya ağdan okunacak
    # img = cv2.imread(image_url)
    # return img
    
    # Şimdilik siyah bir test resmi dönelim
    return np.zeros((1080, 1920, 3), dtype=np.uint8)

def format_detection_output(cls, landing_status, motion_status, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    """
    Görev 1 çıktılarını istenen formata dönüştürür.
    Sınıflar: 0:Taşıt, 1:İnsan, 2:UAP, 3:UAİ
    İniş Durumu: -1:İniş Alanı Değil, 0:Uygun Değil, 1:Uygun
    Hareket Durumu: -1:Taşıt Değil, 0:Hareketsiz, 1:Hareketli
    """
    return {
        "cls": str(cls),
        "landing_status": str(landing_status),
        "motion_status": str(motion_status),
        "top_left_x": float(top_left_x),
        "top_left_y": float(top_left_y),
        "bottom_right_x": float(bottom_right_x),
        "bottom_right_y": float(bottom_right_y)
    }

def format_translation_output(tx, ty, tz):
    """
    Görev 2 çıktılarını istenen formata dönüştürür.
    """
    return {
        "translation_x": float(tx) if not np.isnan(tx) else "NaN",
        "translation_y": float(ty) if not np.isnan(ty) else "NaN",
        "translation_z": float(tz) if not np.isnan(tz) else "NaN"
    }

def format_undefined_object_output(obj_id, top_left_x, top_left_y, bottom_right_x, bottom_right_y):
    """
    Görev 3 çıktılarını istenen formata dönüştürür.
    """
    return {
        "object_id": int(obj_id),
        "top_left_x": float(top_left_x),
        "top_left_y": float(top_left_y),
        "bottom_right_x": float(bottom_right_x),
        "bottom_right_y": float(bottom_right_y)
    }
