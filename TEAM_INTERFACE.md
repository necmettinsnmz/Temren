# Teknofest 2026 Havacılıkta Yapay Zeka - Takım Geliştirme Arayüzü (Interface)

Merhaba Takım!

Projemizin sorunsuz çalışabilmesi ve üç farklı görevin `main.py` üzerinden ana sisteme hatasız entegre edilebilmesi için herkesin geliştirdiği modülün ortak bir formata uyması gerekmektedir. Farklı çıktı tipleri, isimler veya sürümler yaşamamak adına aşağıdaki arayüz formatına harfiyen uymalıyız.

## Ortak Kurallar
- Proje ortamı olarak `venv` kullanılacaktır. Geliştirmelerinizi yaparken sanal ortamı (`venv\Scripts\Activate.ps1`) aktif edin. Kurulumları sadece sanal ortama yapın.
- Python sürümlerimizin çakışmaması için `requirements.txt` dosyasındaki kütüphanelere ve versiyonlara bağlı kalın. Ekstra bir kütüphane kurarsanız (örn. PyTorch) mutlaka gruba bildirin ve `requirements.txt` dosyasını güncelleyin.
- Projede ne aşamada olduğumuzu ve olası hataları izleyebilmek için tüm loglamalar `print()` yerine `utils.logger` kullanılarak yapılacaktır. Herhangi bir aşamada hata alırsanız proje dizinindeki `project.log` dosyasına bakabilirsiniz.

## Sınıf (Class) Arayüz Şablonu
Her görev için oluşturacağınız sınıf (Class) şu yapıya sahip olmalıdır:

```python
class BaseTask:
    def __init__(self):
        # Model yükleme, değişkenleri başlatma işlemleri burada yapılacak.
        pass

    def run(self, frame, **kwargs) -> list:
        # Görüntü (frame) numpy array (OpenCV formatında) olarak gelecektir.
        # Dönüş tipi MUTLAKA JSON serileştirmesine uygun bir Python listesi (list) olmalıdır.
        raise NotImplementedError
```

### Görev 1: Nesne Tespiti (task1_detection.py)
**Girdi:** `frame` (numpy array), opsiyonel `use_sahi` (bool) ve `raw_detections` / `detections` (harici/test enjeksiyonu)
**Çıktı:** Sözlük (dictionary) içeren bir liste. Sözlük elemanları `utils.helper` içerisindeki `format_detection_output` fonksiyonu kullanılarak oluşturulmalıdır.
**Çıktı Formatı:**
```json
[
  {
    "cls": "0", 
    "landing_status": "-1", 
    "motion_status": "1", 
    "top_left_x": 100.5, 
    "top_left_y": 200.0, 
    "bottom_right_x": 300.2, 
    "bottom_right_y": 400.1
  }
]
```

### Görev 2: Pozisyon Tespiti (task2_position.py)
**Girdi:** `frame` (numpy array), opsiyonel `health_status` (int), `translation_x`/`translation_y`/`translation_z` (veya `ref_x`/`ref_y`/`ref_z` GPS referansları), `altitude` (float) ve `camera_type` (`"Termal"`, `"RGB 4K"`, `"RGB 1080p"`)
**Çıktı:** Sözlük içeren bir liste. Sözlük elemanları `utils.helper` içerisindeki `format_translation_output` kullanılarak oluşturulmalıdır.
**Çıktı Formatı:**
```json
[
  {
    "translation_x": 0.5,
    "translation_y": -0.2,
    "translation_z": 0.0
  }
]
```

### Görev 3: Görüntü Eşleme (task3_matching.py)
**Girdi:** `frame` (numpy array), opsiyonel olarak `reference_objects`
**Çıktı:** Sözlük içeren bir liste. Sözlük elemanları `utils.helper` içerisindeki `format_undefined_object_output` kullanılarak oluşturulmalıdır.
**Çıktı Formatı:**
```json
[
  {
    "object_id": 1,
    "top_left_x": 262.87,
    "top_left_y": 734.47,
    "bottom_right_x": 405.2,
    "bottom_right_y": 847.3
  }
]
```

Lütfen kendi görev dosyanızda sadece kendi sınıfınız üzerinde çalışın, `main.py` ve `utils` klasöründeki dosyaları kendi başınıza değiştirmeyin. Tüm modüller hazırlandığında sorunsuz bir entegrasyon için bu dokümana sadık kalmak kritik önem taşımaktadır.
