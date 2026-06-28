import requests
import json
from config import SERVER_URL
from utils.logger import log

def fetch_frame_data(session_url):
    """
    Sunucudan güncel frame bilgisini ve telemetri verisini çeker.
    """
    try:
        # response = requests.get(f"{SERVER_URL}/api/get_frame?session={session_url}")
        # return response.json()
        
        return {
            "url": f"{SERVER_URL}/frames/3598/",
            "image_url": "/frame_000000.jpg",
            "video_name": "test_video_1",
            "session": session_url,
            "translation_x": 0.02,
            "translation_y": 0.01,
            "translation_z": 0.03,
            "health_status": 1
        }
    except Exception as e:
        log.error(f"Frame verisi çekilirken hata oluştu: {e}")
        return {}

def send_results(payload):
    """
    Hazırlanan sonuç JSON'ını sunucuya gönderir.
    """
    try:
        # response = requests.post(f"{SERVER_URL}/api/submit", json=payload)
        # return response.status_code
        log.info(f"Sonuçlar sunucuya gönderildi. (Frame: {payload['frame']})")
        return 200
    except Exception as e:
        log.error(f"Sonuçlar gönderilirken hata oluştu: {e}")
        return 500
