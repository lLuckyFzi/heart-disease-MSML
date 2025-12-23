import time
import random
from prometheus_client import start_http_server, Gauge, Counter

MODEL_HEALTH = Gauge('model_health_status', 'Status kesehatan model (1 untuk Aktif)')
PROCESS_TIME = Gauge('system_process_time_seconds', 'Waktu proses sistem dalam detik')
LOGS_PROCESSED = Counter('total_logs_processed_total', 'Total log yang sudah diproses oleh sistem')

if __name__ == '__main__':
    start_http_server(8001)
    print("Prometheus Exporter tambahan berjalan di http://localhost:8001")
    
    while True:
        MODEL_HEALTH.set(1)
        PROCESS_TIME.set(random.uniform(0.1, 0.5))
        LOGS_PROCESSED.inc()
        time.sleep(5)