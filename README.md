# Teknofest 2026 Havacılıkta Yapay Zeka - Temren

Bu proje, 2026 Teknofest Havacılıkta Yapay Zeka yarışması için hazırlanan taslak mimaridir. Takım üyelerinin görevleri ayrı modüllerde geliştirmesine ve `main.py` üzerinden kolayca sunucuya entegre etmesine olanak tanır.

## Görev Dağılımı
- **Görev 1 (Nesne Tespiti):** `tasks/task1_detection.py`
- **Görev 2 (Pozisyon Kestirimi):** `tasks/task2_position.py`
- **Görev 3 (Görüntü Eşleme):** `tasks/task3_matching.py`

## Kurulum
1. `pip install -r requirements.txt` ile bağımlılıkları yükleyin.
2. `config.py` içerisindeki `SERVER_URL` ve takım ayarlarınızı güncelleyin.
3. Sunucuya bağlanıp sistemi test etmek için `python main.py` komutunu çalıştırın.
