"""
Enhanced Logging Module with Thread-Safe CSV Writing
"""
import threading
import csv
from pathlib import Path
from datetime import datetime


class EnhancedLogger:
    """Thread-safe logging system with UTF-8 encoding"""

    def __init__(self, camera_id, log_dir="logs"):
        self.camera_id = camera_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.detail_log = self.log_dir / f"{camera_id}_detail_{self.session_start}.csv"
        self.alert_log = self.log_dir / f"{camera_id}_alerts_{self.session_start}.csv"
        
        # Thread lock for file operations
        self.log_lock = threading.Lock()
        
        self._init_logs()
        
        self.frame_count = 0
        self.alert_count = 0
        self.total_detections = 0
        self.max_risk = 0.0

    def _init_logs(self):
        with self.log_lock:
            # UTF-8 encoding for Windows compatibility
            with open(self.detail_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'frame_num', 'base_risk', 'final_risk', 'alert_status', 'baseline_mode',
                    'detections', 'chaos_risk', 'panic_risk', 'change_risk', 'deviation_risk',
                    'fps', 'processing_time_ms', 'deviation_score', 'th_low', 'th_high'
                ])

            with open(self.alert_log, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'alert_type', 'final_risk', 'base_risk', 'detections',
                    'trigger_reason', 'duration_seconds', 'deviation_score', 'baseline_mode'
                ])

    def log_frame(self, data):
        with self.log_lock:
            with open(self.detail_log, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    data.get('frame_num', 0),
                    f"{data.get('base_risk', 0):.4f}",
                    f"{data.get('risk_score', 0):.4f}",
                    data.get('alert_status', 'OK'),
                    'ON' if data.get('baseline_mode', False) else 'OFF',
                    data.get('detections', 0),
                    f"{data.get('chaos_risk', 0):.4f}",
                    f"{data.get('panic_risk', 0):.4f}",
                    f"{data.get('change_risk', 0):.4f}",
                    f"{data.get('deviation_risk', 0):.4f}",
                    f"{data.get('fps', 0):.2f}",
                    f"{data.get('processing_time', 0):.2f}",
                    f"{data.get('deviation_score', 0):.2f}",
                    f"{data.get('th_low', 0):.4f}",
                    f"{data.get('th_high', 0):.4f}"
                ])

        self.frame_count += 1
        self.total_detections += data.get('detections', 0)
        risk = data.get('risk_score', 0)
        if risk > self.max_risk:
            self.max_risk = risk

    def log_alert(self, data):
        with self.log_lock:
            with open(self.alert_log, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    data.get('alert_type', 'RISK_ALERT'),
                    f"{data.get('risk_score', 0):.4f}",
                    f"{data.get('base_risk', 0):.4f}",
                    data.get('detections', 0),
                    data.get('trigger_reason', 'Unknown'),
                    f"{data.get('duration', 0):.2f}",
                    f"{data.get('deviation_score', 0):.2f}",
                    'ON' if data.get('baseline_mode', False) else 'OFF'
                ])
        self.alert_count += 1

    def console_log(self, message, level='INFO'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            'INFO': '\033[94m',
            'SUCCESS': '\033[92m',
            'WARNING': '\033[93m',
            'ERROR': '\033[91m',
            'ALERT': '\033[95m',
            'LEARNING': '\033[96m',
            'RESET': '\033[0m'
        }
        color = colors.get(level, colors['INFO'])
        reset = colors['RESET']
        print(f"{color}[{timestamp}] [{self.camera_id}] [{level}] {message}{reset}")
