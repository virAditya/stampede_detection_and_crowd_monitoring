"""
PyQt5 Dashboard for Multi-Camera Monitoring
"""
import sys
import time
import queue
from collections import deque
from pathlib import Path

from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QTextEdit, 
                              QPushButton, QComboBox, QHBoxLayout, QGroupBox, 
                              QGridLayout, QTabWidget)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
import pyqtgraph as pg
import cv2


class MultiCameraDashboard(QWidget):
    def __init__(self, camera_configs, shared_dict, frame_queue, log_queue, stop_event, baseline_mode_dict):
        super().__init__()
        
        self.camera_configs = camera_configs
        self.camera_ids = list(camera_configs.keys())
        self.shared_dict = shared_dict
        self.frame_queue = frame_queue
        self.log_queue = log_queue
        self.stop_event = stop_event
        self.baseline_mode_dict = baseline_mode_dict
        
        self.current_cam = self.camera_ids[0]
        self.risk_history = {cid: deque(maxlen=100) for cid in self.camera_ids}
        self.current_frames = {cid: None for cid in self.camera_ids}
        self.log_messages = deque(maxlen=500)
        self.camera_logs = {cid: deque(maxlen=100) for cid in self.camera_ids}
        
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Multi-Camera Stampede Detection - OPTIMIZED [IMG_SIZE:1120]")
        self.setGeometry(100, 100, 1600, 900)
        
        mainlayout = QVBoxLayout()
        
        # Control bar
        controllayout = QHBoxLayout()
        
        titlelabel = QLabel("🔥 OPTIMIZED Multi-Camera Detection - 5x Faster!")
        titlelabel.setFont(QFont("Arial", 16, QFont.Bold))
        controllayout.addWidget(titlelabel)
        controllayout.addStretch()
        
        # Per-camera baseline toggle button
        self.baseline_toggle_btn = QPushButton(f"{self.current_cam} Baseline OFF")
        self.baseline_toggle_btn.clicked.connect(self.toggle_baseline_mode)
        self.baseline_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        controllayout.addWidget(self.baseline_toggle_btn)
        
        controllayout.addWidget(QLabel("Camera:"))
        self.cam_selector = QComboBox()
        for cam_id, config in self.camera_configs.items():
            display_name = f"{cam_id} - {Path(config['video']).name}"
            self.cam_selector.addItem(display_name, cam_id)
        self.cam_selector.currentIndexChanged.connect(self.change_camera)
        controllayout.addWidget(self.cam_selector)
        
        self.stop_btn = QPushButton("⛔ Stop All")
        self.stop_btn.clicked.connect(self.stop_systems)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        controllayout.addWidget(self.stop_btn)
        
        mainlayout.addLayout(controllayout)
        
        # Tabs
        self.tabs = QTabWidget()
        
        # TAB 1: LIVE MONITORING
        monitor_tab = QWidget()
        monitor_mainlayout = QVBoxLayout()
        
        toplayout = QHBoxLayout()
        
        # Video panel
        leftpanel = QVBoxLayout()
        video_group = QGroupBox("Live Feed")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(960, 540)
        self.video_label.setStyleSheet("border: 3px solid #333; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label)
        video_group.setLayout(video_layout)
        leftpanel.addWidget(video_group)
        
        toplayout.addLayout(leftpanel, 55)
        
        # Right panel
        rightpanel = QVBoxLayout()
        
        # Overview
        overview_group = QGroupBox("All Cameras")
        overview_layout = QVBoxLayout()
        self.camera_status_labels = {}
        for cam_id in self.camera_ids:
            cam_label = QLabel(f"{cam_id}: Initializing...")
            cam_label.setFont(QFont("Courier", 9))
            cam_label.setStyleSheet("padding: 6px; background-color: #f8f9fa; border: 1px solid #dee2e6;")
            self.camera_status_labels[cam_id] = cam_label
            overview_layout.addWidget(cam_label)
        overview_group.setLayout(overview_layout)
        rightpanel.addWidget(overview_group)
        
        # Risk graph
        graph_group = QGroupBox("Risk Trend")
        graph_layout = QVBoxLayout()
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        self.plot_widget.setYRange(0, 1)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_curve = self.plot_widget.plot(pen=pg.mkPen('#dc3545', width=3))
        graph_layout.addWidget(self.plot_widget)
        graph_group.setLayout(graph_layout)
        rightpanel.addWidget(graph_group)
        
        toplayout.addLayout(rightpanel, 45)
        
        monitor_mainlayout.addLayout(toplayout, 60)
        
        # Bottom panel
        bottomlayout = QHBoxLayout()
        
        # Current metrics
        status_group = QGroupBox("Current Camera Metrics")
        status_layout = QGridLayout()
        
        self.risk_label = QLabel("Final Risk: --")
        self.risk_label.setFont(QFont("Arial", 13, QFont.Bold))
        status_layout.addWidget(self.risk_label, 0, 0)
        
        self.alert_label = QLabel("Status: --")
        self.alert_label.setFont(QFont("Arial", 13, QFont.Bold))
        status_layout.addWidget(self.alert_label, 0, 1)
        
        self.base_risk_label = QLabel("Base Risk: --")
        self.base_risk_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.base_risk_label, 1, 0)
        
        self.mode_label = QLabel("Mode: --")
        self.mode_label.setFont(QFont("Arial", 10, QFont.Bold))
        status_layout.addWidget(self.mode_label, 1, 1)
        
        self.count_label = QLabel("Det: 0")
        self.count_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.count_label, 2, 0)
        
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.fps_label, 2, 1)
        
        self.frame_label = QLabel("Frame: --")
        self.frame_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.frame_label, 3, 0)
        
        self.chaos_label = QLabel("Chaos: --")
        self.chaos_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.chaos_label, 3, 1)
        
        self.panic_label = QLabel("Panic: --")
        self.panic_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.panic_label, 4, 0)
        
        self.deviation_label = QLabel("Deviation: --")
        self.deviation_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.deviation_label, 4, 1)
        
        self.deviation_risk_label = QLabel("Dev Risk: --")
        self.deviation_risk_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.deviation_risk_label, 5, 0)
        
        self.th_low_label = QLabel("TH_LOW: --")
        self.th_low_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.th_low_label, 5, 1)
        
        self.th_high_label = QLabel("TH_HIGH: --")
        self.th_high_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.th_high_label, 6, 0)
        
        self.calibrated_label = QLabel("🔄 Learning...")
        self.calibrated_label.setFont(QFont("Arial", 10))
        status_layout.addWidget(self.calibrated_label, 6, 1)
        
        status_group.setLayout(status_layout)
        bottomlayout.addWidget(status_group, 35)
        
        # Camera log
        camera_log_group = QGroupBox("Activity Log")
        camera_log_layout = QVBoxLayout()
        
        camera_log_control = QHBoxLayout()
        self.log_title = QLabel(f"Camera: {self.current_cam}")
        self.log_title.setFont(QFont("Arial", 10, QFont.Bold))
        camera_log_control.addWidget(self.log_title)
        camera_log_control.addStretch()
        
        clear_camera_log_btn = QPushButton("Clear")
        clear_camera_log_btn.clicked.connect(self.clear_camera_logs)
        camera_log_control.addWidget(clear_camera_log_btn)
        camera_log_layout.addLayout(camera_log_control)
        
        self.camera_log_viewer = QTextEdit()
        self.camera_log_viewer.setReadOnly(True)
        self.camera_log_viewer.setFont(QFont("Courier", 8))
        self.camera_log_viewer.setStyleSheet("""
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 2px solid #555;
            padding: 5px;
        """)
        self.camera_log_viewer.setMaximumHeight(180)
        camera_log_layout.addWidget(self.camera_log_viewer)
        camera_log_group.setLayout(camera_log_layout)
        bottomlayout.addWidget(camera_log_group, 35)
        
        monitor_mainlayout.addLayout(bottomlayout, 40)
        monitor_tab.setLayout(monitor_mainlayout)
        
        # TAB 2: ALL LOGS
        logs_tab = QWidget()
        logs_layout = QVBoxLayout()
        
        log_control = QHBoxLayout()
        log_control.addWidget(QLabel("System Logs (All Cameras)"))
        log_control.addStretch()
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.clicked.connect(self.clear_all_logs)
        log_control.addWidget(clear_all_btn)
        logs_layout.addLayout(log_control)
        
        self.all_log_viewer = QTextEdit()
        self.all_log_viewer.setReadOnly(True)
        self.all_log_viewer.setFont(QFont("Courier", 10))
        self.all_log_viewer.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4;")
        logs_layout.addWidget(self.all_log_viewer)
        logs_tab.setLayout(logs_layout)
        
        self.tabs.addTab(monitor_tab, "🎥 Live Monitoring")
        self.tabs.addTab(logs_tab, "📋 All Logs")
        
        mainlayout.addWidget(self.tabs)
        self.setLayout(mainlayout)
        
        # Timers
        self.data_timer = QTimer()
        self.data_timer.timeout.connect(self.update_data)
        self.data_timer.start(100)
        
        self.frame_timer = QTimer()
        self.frame_timer.timeout.connect(self.update_frames)
        self.frame_timer.start(50)
        
        self.log_timer = QTimer()
        self.log_timer.timeout.connect(self.update_logs)
        self.log_timer.start(200)
        
        # Button state synchronization timer
        self.button_sync_timer = QTimer()
        self.button_sync_timer.timeout.connect(self.sync_button_state)
        self.button_sync_timer.start(1000)

    def toggle_baseline_mode(self):
        """Toggle baseline mode for CURRENT camera only"""
        if self.current_cam not in self.shared_dict:
            print(f"[GUI] No data for {self.current_cam}")
            return
        
        data = self.shared_dict[self.current_cam]
        can_use = data.get('can_use_baseline', False)
        total_frames = data.get('total_baseline_frames', 0)
        
        if not can_use:
            print(f"[GUI] {self.current_cam} baseline locked - need 5000 frames (currently {total_frames})")
            return
        
        cam_baseline_dict = self.baseline_mode_dict[self.current_cam]
        temp_dict = dict(cam_baseline_dict)
        temp_dict['enabled'] = not temp_dict.get('enabled', False)
        cam_baseline_dict.clear()
        cam_baseline_dict.update(temp_dict)
        
        new_state = temp_dict['enabled']
        if new_state:
            self.baseline_toggle_btn.setText(f"{self.current_cam} Baseline ON")
            self.baseline_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
            print(f"[GUI] {self.current_cam} Baseline mode ENABLED")
        else:
            self.baseline_toggle_btn.setText(f"{self.current_cam} Baseline OFF")
            self.baseline_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
            print(f"[GUI] {self.current_cam} Baseline mode DISABLED")

    def sync_button_state(self):
        """Sync button appearance with actual baseline mode"""
        if self.current_cam not in self.baseline_mode_dict:
            return
        
        cam_baseline_dict = self.baseline_mode_dict[self.current_cam]
        current_state = cam_baseline_dict.get('enabled', False)
        
        if current_state:
            self.baseline_toggle_btn.setText(f"{self.current_cam} Baseline ON")
            self.baseline_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #28a745;
                    color: white;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #218838;
                }
            """)
        else:
            self.baseline_toggle_btn.setText(f"{self.current_cam} Baseline OFF")
            self.baseline_toggle_btn.setStyleSheet("""
                QPushButton {
                    background-color: #6c757d;
                    color: white;
                    font-weight: bold;
                    padding: 10px 20px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #5a6268;
                }
            """)
        
        if self.current_cam in self.shared_dict:
            can_use = self.shared_dict[self.current_cam].get('can_use_baseline', False)
            total_frames = self.shared_dict[self.current_cam].get('total_baseline_frames', 0)
            
            if not can_use:
                remaining = max(0, 5000 - total_frames)
                self.baseline_toggle_btn.setToolTip(f"{self.current_cam} Locked - {remaining} frames remaining")
                self.baseline_toggle_btn.setEnabled(False)
            else:
                self.baseline_toggle_btn.setToolTip(f"{self.current_cam}: Toggle baseline detection mode")
                self.baseline_toggle_btn.setEnabled(True)

    def change_camera(self):
        self.current_cam = self.cam_selector.currentData()
        self.log_title.setText(f"Camera: {self.current_cam}")
        self.update_camera_log_display()
        self.sync_button_state()

    def stop_systems(self):
        print("[GUI] Stopping...")
        self.stop_event.set()
        self.close()

    def clear_camera_logs(self):
        self.camera_logs[self.current_cam].clear()
        self.camera_log_viewer.clear()

    def clear_all_logs(self):
        self.all_log_viewer.clear()
        self.log_messages.clear()

    def update_camera_log_display(self):
        self.camera_log_viewer.clear()
        for log_msg in self.camera_logs[self.current_cam]:
            timestamp = log_msg['timestamp']
            message = log_msg['message']
            level = log_msg['level']
            
            if level == "ALERT":
                color = "#ff6b6b"
            elif level == "WARNING":
                color = "#ffd93d"
            elif level == "SUCCESS":
                color = "#6bcf7f"
            else:
                color = "#e0e0e0"
            
            log_line = f'<span style="color: {color}">{timestamp} {message}</span><br>'
            self.camera_log_viewer.append(log_line)
        
        scrollbar = self.camera_log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_data(self):
        if self.stop_event.is_set():
            self.close()
            return
        
        if self.current_cam in self.shared_dict:
            data = self.shared_dict[self.current_cam]
            score = data.get('risk_score', 0)
            base_risk = data.get('base_risk', 0)
            alert = data.get('alert', False)
            count = data.get('count', 0)
            fps = data.get('fps', 0)
            frame_num = data.get('frame_num', 0)
            chaos = data.get('chaos_risk', 0)
            panic = data.get('panic_risk', 0)
            deviation = data.get('deviation_score', 0)
            deviation_risk = data.get('deviation_risk', 0)
            th_low = data.get('th_low', 0)
            th_high = data.get('th_high', 0)
            calibrated = data.get('calibrated', False)
            baseline_mode = data.get('baseline_mode', False)
            
            self.risk_history[self.current_cam].append(score)
            
            self.risk_label.setText(f"Final Risk: {score:.3f}")
            self.base_risk_label.setText(f"Base Risk: {base_risk:.3f}")
            self.count_label.setText(f"Det: {count}")
            self.fps_label.setText(f"FPS: {fps:.1f}")
            self.frame_label.setText(f"Frame: {frame_num}")
            self.chaos_label.setText(f"Chaos: {chaos:.3f}")
            self.panic_label.setText(f"Panic: {panic:.3f}")
            self.deviation_label.setText(f"Deviation: {deviation:.2f}σ")
            self.deviation_risk_label.setText(f"Dev Risk: {deviation_risk:.3f}")
            self.th_low_label.setText(f"TH_LOW: {th_low:.4f}")
            self.th_high_label.setText(f"TH_HIGH: {th_high:.4f}")
            
            if baseline_mode:
                self.mode_label.setText("Mode: BASELINE ON")
                self.mode_label.setStyleSheet("color: green; font-weight: bold;")
            else:
                self.mode_label.setText("Mode: BASELINE OFF")
                self.mode_label.setStyleSheet("color: orange; font-weight: bold;")
            
            if calibrated:
                self.calibrated_label.setText("✅ Calibrated")
                self.calibrated_label.setStyleSheet("color: green;")
            else:
                total_frames = data.get('total_baseline_frames', 0)
                progress = min(total_frames, 200)
                percent = int((progress / 200) * 100)
                self.calibrated_label.setText(f"🔄 Learning... {percent}%")
                self.calibrated_label.setStyleSheet("color: orange;")
            
            if alert:
                self.alert_label.setText("🚨 ALERT")
                self.alert_label.setStyleSheet("color: red; font-weight: bold;")
                self.risk_label.setStyleSheet("color: red; font-weight: bold;")
            else:
                self.alert_label.setText("✅ NORMAL")
                self.alert_label.setStyleSheet("color: green; font-weight: bold;")
                self.risk_label.setStyleSheet("color: green; font-weight: bold;")
            
            data_list = list(self.risk_history[self.current_cam])
            if data_list:
                self.plot_curve.setData(data_list)
        
        # Update all cameras overview
        for cam_id in self.camera_ids:
            if cam_id in self.shared_dict:
                data = self.shared_dict[cam_id]
                score = data.get('risk_score', 0)
                base_risk = data.get('base_risk', 0)
                alert = data.get('alert', False)
                count = data.get('count', 0)
                fps = data.get('fps', 0)
                baseline_mode = data.get('baseline_mode', False)
                
                mode_str = "[OPT][B:ON]" if baseline_mode else "[OPT][B:OFF]"
                status_text = f"{cam_id} {mode_str}: Risk={score:.2f}(B={base_risk:.2f}) Det={count} FPS={fps:.1f}"
                status_text += " 🚨 ALERT" if alert else " ✅ OK"
                
                self.camera_status_labels[cam_id].setText(status_text)
                
                if alert:
                    self.camera_status_labels[cam_id].setStyleSheet("""
                        padding: 6px;
                        background-color: #f8d7da;
                        border: 2px solid #dc3545;
                        font-weight: bold;
                    """)
                else:
                    self.camera_status_labels[cam_id].setStyleSheet("""
                        padding: 6px;
                        background-color: #d4edda;
                        border: 1px solid #28a745;
                    """)

    def update_logs(self):
        new_logs = []
        while not self.log_queue.empty():
            try:
                log_msg = self.log_queue.get_nowait()
                new_logs.append(log_msg)
                self.log_messages.append(log_msg)
                
                cam_id = log_msg['camera_id']
                self.camera_logs[cam_id].append(log_msg)
                
                timestamp = log_msg['timestamp']
                cam_id_display = log_msg['camera_id']
                message = log_msg['message']
                level = log_msg['level']
                
                if level == "ALERT":
                    color = "#ff6b6b"
                elif level == "WARNING":
                    color = "#ffd93d"
                elif level == "SUCCESS":
                    color = "#6bcf7f"
                else:
                    color = "#d4d4d4"
                
                log_line = f'<span style="color: {color}">{timestamp} [{cam_id_display}] {message}</span><br>'
                self.all_log_viewer.append(log_line)
                
            except queue.Empty:
                break
            except Exception as e:
                print(f"Log queue error: {e}")
                break
        
        if any(log['camera_id'] == self.current_cam for log in new_logs):
            self.update_camera_log_display()
        
        scrollbar = self.all_log_viewer.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_frames(self):
        latest_frames = {}
        count = 0
        while not self.frame_queue.empty() and count < 10:
            try:
                cam_id, frame = self.frame_queue.get_nowait()
                latest_frames[cam_id] = frame
                count += 1
            except queue.Empty:
                break
            except Exception as e:
                print(f"Frame queue error: {e}")
                break
        
        self.current_frames.update(latest_frames)
        
        if self.current_cam in self.current_frames and self.current_frames[self.current_cam] is not None:
            frame = self.current_frames[self.current_cam]
            try:
                frame_resized = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_LINEAR)
                img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                h, w, ch = img.shape
                bytes_per_line = ch * w
                qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_img))
            except:
                pass


def run_gui_dashboard(shared_dict, frame_queue, log_queue, stop_event, baseline_mode_dict, camera_configs):
    """Complete GUI Dashboard with per-camera baseline controls"""
    print("[GUI] Starting Enhanced Dashboard with Per-Camera Baseline Control...")
    
    try:
        app = QApplication(sys.argv)
        dashboard = MultiCameraDashboard(camera_configs, shared_dict, frame_queue, log_queue, stop_event, baseline_mode_dict)
        dashboard.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"[GUI] ERROR: {e}")
        import traceback
        traceback.print_exc()
