import logging
import sys

def setup_logger(name="teknofest_logger"):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Log dosyası ayarları
        file_handler = logging.FileHandler('project.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Konsol ayarları
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

# Global logger objesi
log = setup_logger()
