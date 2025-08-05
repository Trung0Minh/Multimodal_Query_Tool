# main.py
import os
import sys
import shutil
import atexit
import signal

# add project folder to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.settings import settings
from utils.logger import logger
from app import create_and_run_app
from app import shared_qdrant_client

GLOBAL_QDRANT_CLIENT = shared_qdrant_client

def cleanup():
    logger.info("--- Starting cleanup process ---")

    # --- Step 1: Close Qdrant connection ---
    # release file .lock
    global GLOBAL_QDRANT_CLIENT
    if GLOBAL_QDRANT_CLIENT:
        try:
            logger.info("Closing Qdrant client connection...")
            GLOBAL_QDRANT_CLIENT.close()
            logger.success("Qdrant client closed successfully.")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")
    
    # Step 2: Clean up "qdrant_data" folder
    qdrant_db_path = os.path.join(settings.DATA_DIR, "qdrant_data")
    if os.path.exists(qdrant_db_path) and os.path.isdir(qdrant_db_path):
        try:
            # try many times just in case
            import time
            retries = 3
            for i in range(retries):
                try:
                    shutil.rmtree(qdrant_db_path)
                    logger.success(f"Successfully cleaned up Qdrant data at: {qdrant_db_path}")
                    break
                except OSError as e:
                    if i < retries - 1:
                        logger.warning(f"Cleanup attempt {i+1} failed: {e}. Retrying in 1 second...")
                        time.sleep(1)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error cleaning up Qdrant data after retries: {e}")
    else:
        logger.info("Qdrant data directory not found, skipping cleanup.")

    # Step 3: Clean up "raw" folder
    raw_data_path = settings.RAW_DATA_DIR
    if os.path.exists(raw_data_path) and os.path.isdir(raw_data_path):
        try:
            for item in os.listdir(raw_data_path):
                item_path = os.path.join(raw_data_path, item)
                if os.path.isdir(item_path): shutil.rmtree(item_path)
                elif os.path.isfile(item_path): os.remove(item_path)
            logger.success(f"Successfully cleaned up raw data directory: {raw_data_path}")
        except Exception as e:
            logger.error(f"Error cleaning up raw data: {e}")
    else:
        logger.info("Raw data directory not found, skipping cleanup.")
        
    # Step 4: Clean up "processed/chunks" folder
    chunks_data_path = settings.CHUNKS_DIR
    if os.path.exists(chunks_data_path) and os.path.isdir(chunks_data_path):
        try:
            for item in os.listdir(chunks_data_path):
                item_path = os.path.join(chunks_data_path, item)
                if os.path.isdir(item_path): shutil.rmtree(item_path)
                elif os.path.isfile(item_path): os.remove(item_path)
            logger.success(f"Successfully cleaned up processed/chunks data directory: {chunks_data_path}")
        except Exception as e:
            logger.error(f"Error cleaning up processed/chunks data: {e}")
    else:
        logger.info("processed/chunks data directory not found, skipping cleanup.")
    
    logger.info("--- Cleanup process finished ---")

def signal_handler(sig, frame):
    """
    Xử lý tín hiệu ngắt (Ctrl+C) để thoát chương trình một cách an toàn.
    """
    logger.warning("\nCtrl+C detected. Shutting down the application.")
    # atexit sẽ tự động gọi cleanup()
    sys.exit(0)

def main():
    # Đăng ký hàm cleanup để được gọi khi chương trình thoát bình thường hoặc có lỗi
    atexit.register(cleanup)
    
    # Đăng ký hàm xử lý tín hiệu cho Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info("--- Starting Multimedia RAG Assistant ---")
    
    demo = create_and_run_app()
    
    print("\n" + "="*50)
    print("     Application is running. Press Ctrl+C to exit.     ")
    print("     Cleanup will be performed upon exit.               ")
    print("="*50 + "\n")
    
    demo.launch()

if __name__ == "__main__":
    main()