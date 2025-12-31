"""
Continuous Baseline Learning Module - Deadlock-Free Implementation
"""
import threading
import json
import numpy as np
from pathlib import Path
from datetime import datetime


class ContinuousBaselineLearner:
    """Camera-specific continuous baseline learning - DEADLOCK-FREE VERSION"""
    
    def __init__(self, camera_id, baseline_dir="baselines"):
        self.camera_id = camera_id
        # Use absolute path to avoid working directory issues
        script_dir = Path(__file__).parent.parent.resolve()
        self.baseline_dir = script_dir / baseline_dir
        self.baseline_dir.mkdir(exist_ok=True)
        self.baseline_file = self.baseline_dir / f"{camera_id}_baseline.json"
        
        self.baseline_data = {
            'camera_id': camera_id,
            'calibrated': False,
            'total_frames_processed': 0,
            'last_update_frame': 0,
            'mean_risk': 0.0,
            'std_risk': 0.0,
            'mean_density': 0.0,
            'std_density': 0.0,
            'mean_flow_speed': 0.0,
            'std_flow_speed': 0.0,
            'mean_detections': 0.0,
            'std_detections': 0.0,
            'th_low': 0.20,
            'th_high': 0.40,
            'last_update_timestamp': None
        }
        
        self.load_baseline()
        self.learning_buffer = {'risk': [], 'density': [], 'flow_speed': [], 'detections': []}
        self.lock = threading.Lock()
        
        # Separate lock for file I/O operations to prevent deadlock
        self.file_lock = threading.Lock()
    
    def load_baseline(self):
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.baseline_data.update(loaded)
                    print(f"[{self.camera_id}] 📂 Loaded baseline: {self.baseline_data['total_frames_processed']} frames")
                    print(f"[{self.camera_id}] Baseline location: {self.baseline_file.absolute()}")
                    return True
            except Exception as e:
                print(f"[{self.camera_id}] Error loading baseline: {e}")
        return False
    
    def save_baseline(self):
        """DEADLOCK-FREE: Save baseline without holding the main lock"""
        # Make a copy of data WITHOUT holding the main lock
        with self.lock:
            data_copy = dict(self.baseline_data)
        
        # File I/O with separate lock (doesn't block main processing)
        try:
            with self.file_lock:
                with open(self.baseline_file, "w", encoding="utf-8") as f:
                    json.dump(data_copy, f, indent=2)
            print(f"[{self.camera_id}] 💾 Baseline saved (Frame {data_copy['total_frames_processed']})")
        except Exception as e:
            print(f"[{self.camera_id}] Error saving baseline: {e}")
    
    def add_sample(self, risk_score, density, flow_speed, detections):
        """Add sample to learning buffer - NON-BLOCKING"""
        try:
            with self.lock:
                self.learning_buffer['risk'].append(risk_score)
                self.learning_buffer['density'].append(density)
                self.learning_buffer['flow_speed'].append(flow_speed)
                self.learning_buffer['detections'].append(detections)
                self.baseline_data['total_frames_processed'] += 1
        except Exception as e:
            print(f"[{self.camera_id}] Error adding sample: {e}")
    
    def should_update_baseline(self):
        """Determine if baseline should be updated - NON-BLOCKING"""
        try:
            with self.lock:
                total_frames = self.baseline_data['total_frames_processed']
                last_update = self.baseline_data['last_update_frame']
                buffer_size = len(self.learning_buffer['risk'])
            
            # Initial learning phase: update every 200 frames
            if total_frames < 5000:
                return (total_frames - last_update) >= 200 and buffer_size >= 200
            # Continuous learning phase: update every 2000 frames
            else:
                return (total_frames - last_update) >= 2000 and buffer_size >= 2000
        except Exception as e:
            print(f"[{self.camera_id}] Error checking update: {e}")
            return False
    
    def update_baseline(self):
        """DEADLOCK-FREE: Update baseline statistics"""
        try:
            # Step 1: Copy buffer data WITHOUT holding lock for long
            with self.lock:
                if len(self.learning_buffer['risk']) < 10:
                    print(f"[{self.camera_id}] ⚠ Insufficient samples for update")
                    return False
                
                # Copy buffer data
                buffer_copy = {
                    'risk': self.learning_buffer['risk'].copy(),
                    'density': self.learning_buffer['density'].copy(),
                    'flow_speed': self.learning_buffer['flow_speed'].copy(),
                    'detections': self.learning_buffer['detections'].copy()
                }
                
                # Clear buffer immediately
                self.learning_buffer = {'risk': [], 'density': [], 'flow_speed': [], 'detections': []}
            
            # Step 2: Compute statistics WITHOUT holding lock (CPU-intensive work)
            mean_risk = float(np.mean(buffer_copy['risk']))
            std_risk = float(np.std(buffer_copy['risk']))
            mean_density = float(np.mean(buffer_copy['density']))
            std_density = float(np.std(buffer_copy['density']))
            mean_flow_speed = float(np.mean(buffer_copy['flow_speed']))
            std_flow_speed = float(np.std(buffer_copy['flow_speed']))
            mean_detections = float(np.mean(buffer_copy['detections']))
            std_detections = float(np.std(buffer_copy['detections']))
            
            # Update thresholds
            std_risk_safe = max(std_risk, 0.01)
            th_low = max(0.05, mean_risk - 0.5 * std_risk_safe)
            th_high = min(0.95, mean_risk + 2.5 * std_risk_safe)
            
            # Step 3: Update baseline data (quick operation with lock)
            with self.lock:
                self.baseline_data['mean_risk'] = mean_risk
                self.baseline_data['std_risk'] = std_risk
                self.baseline_data['mean_density'] = mean_density
                self.baseline_data['std_density'] = std_density
                self.baseline_data['mean_flow_speed'] = mean_flow_speed
                self.baseline_data['std_flow_speed'] = std_flow_speed
                self.baseline_data['mean_detections'] = mean_detections
                self.baseline_data['std_detections'] = std_detections
                self.baseline_data['th_low'] = th_low
                self.baseline_data['th_high'] = th_high
                
                # Mark as calibrated after first update
                was_calibrated = self.baseline_data['calibrated']
                if not was_calibrated and self.baseline_data['total_frames_processed'] >= 200:
                    self.baseline_data['calibrated'] = True
                
                # Update metadata
                total_frames = self.baseline_data['total_frames_processed']
                self.baseline_data['last_update_frame'] = total_frames
                self.baseline_data['last_update_timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Step 4: Print success message (outside lock)
            if not was_calibrated:
                print(f"\n{'='*60}")
                print(f"[{self.camera_id}] ✅ BASELINE INITIALLY CALIBRATED")
                print(f"{'='*60}")
            
            print(f"\n{'='*60}")
            print(f"[{self.camera_id}] 🔄 BASELINE UPDATED")
            print(f"{'='*60}")
            print(f"  Total frames: {total_frames}")
            print(f"  Samples used: {len(buffer_copy['risk'])}")
            print(f"  Mean Risk: {mean_risk:.4f} ± {std_risk:.4f}")
            print(f"  TH_LOW: {th_low:.4f}")
            print(f"  TH_HIGH: {th_high:.4f}")
            print(f"{'='*60}\n")
            
            # Step 5: Save to file (asynchronous, doesn't block)
            self.save_baseline()
            return True
            
        except Exception as e:
            print(f"[{self.camera_id}] ERROR in update_baseline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_thresholds(self):
        """Thread-safe threshold getter"""
        try:
            with self.lock:
                return self.baseline_data['th_low'], self.baseline_data['th_high']
        except:
            return 0.20, 0.40
    
    def is_calibrated(self):
        """Thread-safe calibration status"""
        try:
            with self.lock:
                return self.baseline_data['calibrated']
        except:
            return False
    
    def get_total_frames(self):
        """Thread-safe frame count"""
        try:
            with self.lock:
                return self.baseline_data['total_frames_processed']
        except:
            return 0
    
    def can_use_baseline_mode(self):
        """Baseline mode only available after 5000 frames"""
        return self.get_total_frames() >= 5000
    
    def get_deviation_score(self, current_risk):
        """Thread-safe deviation score calculation"""
        try:
            with self.lock:
                if not self.baseline_data['calibrated'] or self.baseline_data['std_risk'] < 0.001:
                    return 0.0
                deviation = (current_risk - self.baseline_data['mean_risk']) / self.baseline_data['std_risk']
                return float(deviation)
        except:
            return 0.0
    
    def get_deviation_risk(self, current_risk):
        """Thread-safe deviation risk calculation"""
        try:
            deviation = self.get_deviation_score(current_risk)
            deviation_risk = np.tanh(abs(deviation) / 3.0)
            return float(deviation_risk)
        except:
            return 0.0
