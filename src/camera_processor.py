"""
Camera Processing Module - Optimized Risk Detection
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from math import pi
from collections import deque
import time
import threading
import queue
from pathlib import Path

from baseline_learner import ContinuousBaselineLearner
from logger import EnhancedLogger
from video_reader import AsyncVideoReader


def run_risk_detection_camera(camera_id, video_path, model_path, shared_dict, frame_queue, 
                               stop_event, log_queue, baseline_mode_dict):
    """
    OPTIMIZED multi-threaded camera process
    """
    print(f"[{camera_id}] Starting OPTIMIZED Risk Detection System...")

    try:
        baseline = ContinuousBaselineLearner(camera_id)
        logger = EnhancedLogger(camera_id)
        logger.console_log("Logger initialized", "SUCCESS")

        # OPTIMIZED Configuration
        GRID_W, GRID_H = 20, 12
        IMG_SIZE = 1120  # High quality: 35×32
        CONF_THRES = 0.30
        IOU_THRES = 0.50
        ALPHA = 0.60
        TARGET_FPS = 2.0  # Realistic target
        MAX_DETECTIONS = 300
        DEVIATION_WEIGHT = 0.10
        BASE_WEIGHT = 0.90
        FRAME_LOG_INTERVAL = 10
        OPTICAL_FLOW_WIDTH = 320   # Optimized: smaller, faster
        OPTICAL_FLOW_HEIGHT = 180  # Optimized: smaller, faster

        device = 0 if torch.cuda.is_available() else 'cpu'
        logger.console_log(f"Using device: {device}", "INFO")

        model = YOLO(model_path)
        logger.console_log("Model loaded", "SUCCESS")

        # Start async video reader
        video_reader = AsyncVideoReader(video_path, frame_buffer_size=30)
        video_reader.start()
        logger.console_log("Async video reader started", "SUCCESS")
        
        # Get video properties
        cap_temp = cv2.VideoCapture(video_path)
        W = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
        SRC_FPS = cap_temp.get(cv2.CAP_PROP_FPS) or 25.0
        cap_temp.release()

        logger.console_log(f"Video: {W}x{H} @ {SRC_FPS:.1f}fps", "INFO")
        
        if baseline.is_calibrated():
            TH_LOW_ADAPTIVE, TH_HIGH_ADAPTIVE = baseline.get_thresholds()
            logger.console_log(f"✓ Baseline loaded: LOW={TH_LOW_ADAPTIVE:.4f}, HIGH={TH_HIGH_ADAPTIVE:.4f}", "SUCCESS")
        else:
            TH_LOW_ADAPTIVE, TH_HIGH_ADAPTIVE = 0.20, 0.40
            logger.console_log(f"⚠ No baseline found, starting fresh", "WARNING")

        TH_LOW_FIXED = 0.20
        TH_HIGH_FIXED = 0.40

        # OPTIMIZED: Force minimum skip of 2 frames
        MIN_STEP, MAX_STEP = 2, max(2, int(round(SRC_FPS / TARGET_FPS)))
        STEP = MAX_STEP

        prev_gray = None
        ema_cell = None
        prev_global = None
        alert_state = "OK"
        prev_count = None

        frame_idx = 0
        optical_flow_counter = 0
        detection_counter = 0

        last_speed_var = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        last_dir_entropy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        last_avg_speed = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        last_direction_coherence = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        last_speed_magnitude = np.zeros((GRID_H, GRID_W), dtype=np.float32)
        last_centers = []
        last_count = 0
        
        R_buffer = deque(maxlen=5)

        last_frame_time = time.time()
        alert_start_time = None
        
        # YOLO inference queue for async processing
        inference_queue = queue.Queue(maxsize=2)
        inference_lock = threading.Lock()
        latest_detection_result = {'centers': [], 'count': 0}
        
        def yolo_inference_worker():
            """Separate thread for YOLO inference"""
            while not stop_event.is_set():
                try:
                    frame_data = inference_queue.get(timeout=0.5)
                    if frame_data is None:
                        continue
                    
                    frame = frame_data['frame']
                    res = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRES, iou=IOU_THRES, 
                                      device=device, verbose=False, max_det=2000)[0]
                    
                    centers = []
                    count = 0
                    if hasattr(res, "obb") and res.obb is not None:
                        count = len(res.obb) if hasattr(res.obb, "__len__") else 0
                        if hasattr(res.obb, "xyxyxyxy") and res.obb.xyxyxyxy is not None:
                            polys = res.obb.xyxyxyxy.cpu().numpy()
                            if len(polys) > 0:
                                centers = polys.reshape(-1, 4, 2).mean(axis=1).tolist()
                    elif hasattr(res, "boxes") and res.boxes is not None:
                        count = len(res.boxes)
                        if hasattr(res.boxes, "xywh") and res.boxes.xywh is not None:
                            xywh = res.boxes.xywh.cpu().numpy()
                            if len(xywh) > 0:
                                centers = xywh[:, :2].tolist()
                    
                    with inference_lock:
                        latest_detection_result['centers'] = centers
                        latest_detection_result['count'] = count
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.console_log(f"Inference error: {e}", "ERROR")
        
        # Start YOLO inference thread
        inference_thread = threading.Thread(target=yolo_inference_worker, daemon=True)
        inference_thread.start()
        logger.console_log("YOLO inference thread started", "SUCCESS")

        # Utility functions
        def grid_bins(W, H, gw, gh):
            return W / gw, H / gh

        def assign_to_grid(centers, W, H, gw, gh):
            cell_w, cell_h = grid_bins(W, H, gw, gh)
            counts = np.zeros((gh, gw), dtype=np.float32)
            for cx, cy in centers:
                j = int(cx / cell_w)
                i = int(cy / cell_h)
                if 0 <= i < gh and 0 <= j < gw:
                    counts[i, j] += 1.0
            return counts

        def flow_features_direct(flow, gw, gh):
            """OPTIMIZED: Direct computation without threading overhead"""
            Hf, Wf = flow.shape[:2]
            cell_w, cell_h = Wf / gw, Hf / gh
            
            fx = flow[..., 0]
            fy = flow[..., 1]
            speed = np.sqrt(fx**2 + fy**2)
            angle = np.arctan2(fy, fx)
            
            speed_var = np.zeros((gh, gw), dtype=np.float32)
            dir_entropy = np.zeros((gh, gw), dtype=np.float32)
            
            for i in range(gh):
                y0 = int(i * cell_h)
                y1 = int(min((i + 1) * cell_h, Hf))
                for j in range(gw):
                    x0 = int(j * cell_w)
                    x1 = int(min((j + 1) * cell_w, Wf))
                    
                    s = speed[y0:y1, x0:x1].ravel()
                    a = angle[y0:y1, x0:x1].ravel()
                    
                    if s.size < 8:
                        continue
                        
                    speed_var[i, j] = s.var()
                    
                    hist, _ = np.histogram(a, bins=np.linspace(-np.pi, np.pi, 9))
                    p = hist.astype(np.float32) / (hist.sum() + 1e-9)
                    H_raw = -np.sum(p * np.log(p + 1e-9))
                    H_max = np.log(8 + 1e-9)
                    dir_entropy[i, j] = H_raw / (H_max + 1e-9)
            
            return speed_var, dir_entropy

        def panic_indicators(flow, gw, gh):
            """Panic indicators with optimized computation"""
            Hf, Wf = flow.shape[:2]
            cell_w, cell_h = grid_bins(Wf, Hf, gw, gh)
            avg_speed = np.zeros((gh, gw), dtype=np.float32)
            direction_coherence = np.zeros((gh, gw), dtype=np.float32)
            speed_magnitude = np.zeros((gh, gw), dtype=np.float32)
            
            fx = flow[..., 0]
            fy = flow[..., 1]
            speed = np.sqrt(fx**2 + fy**2)
            angle = np.arctan2(fy, fx)
            
            for i in range(gh):
                y0 = int(i * cell_h)
                y1 = int(min((i + 1) * cell_h, Hf))
                for j in range(gw):
                    x0 = int(j * cell_w)
                    x1 = int(min((j + 1) * cell_w, Wf))
                    s = speed[y0:y1, x0:x1].ravel()
                    a = angle[y0:y1, x0:x1].ravel()
                    if s.size < 8:
                        continue
                    avg_speed[i, j] = np.mean(s)
                    speed_magnitude[i, j] = np.percentile(s, 75)
                    hist, _ = np.histogram(a, bins=np.linspace(-pi, pi, 9))
                    p = hist.astype(np.float32) / (hist.sum() + 1e-9)
                    entropy = -np.sum(p * np.log(p + 1e-9))
                    max_entropy = np.log(8)
                    direction_coherence[i, j] = 1.0 - (entropy / (max_entropy + 1e-9))
            
            return avg_speed, direction_coherence, speed_magnitude

        def minmax01(x, eps=1e-8):
            x_min = np.min(x)
            x_max = np.max(x)
            if x_max - x_min < eps:
                return np.zeros_like(x)
            return (x - x_min) / (x_max - x_min + eps)

        def ema_update(prev, cur, alpha=0.3):
            if prev is None:
                return cur.copy()
            return alpha * cur + (1 - alpha) * prev

        def color_map(value):
            v = np.clip(value, 0.0, 1.0)
            r = int(255 * v)
            g = int(255 * (1.0 - abs(v - 0.5) * 2.0))
            b = int(255 * (1.0 - v))
            return (b, g, r)

        logger.console_log(f"Starting OPTIMIZED video processing: {Path(video_path).name}", "INFO")

        # Main processing loop
        while not stop_event.is_set():
            frame_start_time = time.time()

            # Get frame from async reader
            frame = video_reader.get_frame(timeout=1.0)
            if frame is None:
                continue

            if frame_idx % STEP != 0:
                frame_idx += 1
                continue

            # OPTIMIZED: YOLO inference every 3rd frame
            detection_counter += 1
            if detection_counter % 3 == 0:
                try:
                    inference_queue.put_nowait({'frame': frame})
                except queue.Full:
                    pass
            
            # Get latest detection result
            with inference_lock:
                centers = latest_detection_result['centers'].copy()
                count = latest_detection_result['count']
            
            if not centers:
                centers = last_centers
                count = last_count
            else:
                last_centers = centers
                last_count = count

            # Risk calculations
            if prev_count is None:
                change_risk = 0.0
            else:
                detection_change = abs(count - prev_count) / max(prev_count, 1)
                change_risk = min(detection_change, 1.0)
                change_risk = np.clip(change_risk * 5.0, 0, 1)
            prev_count = count

            count_risk = min(count / MAX_DETECTIONS, 1.0)
            count_risk = np.clip(count_risk * 4.0, 0, 1)

            density = assign_to_grid(centers, W, H, GRID_W, GRID_H)

            # OPTIMIZED: Optical flow every 3rd frame at reduced resolution
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            optical_flow_counter += 1
            compute_optical_flow = (optical_flow_counter % 3 == 0)
            
            if prev_gray is not None and compute_optical_flow:
                prev_gray_small = cv2.resize(prev_gray, (OPTICAL_FLOW_WIDTH, OPTICAL_FLOW_HEIGHT))
                gray_small = cv2.resize(gray, (OPTICAL_FLOW_WIDTH, OPTICAL_FLOW_HEIGHT))
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )
                
                # OPTIMIZED: Direct computation (no threading overhead)
                speed_var, dir_entropy = flow_features_direct(flow, GRID_W, GRID_H)
                avg_speed, direction_coherence, speed_magnitude = panic_indicators(flow, GRID_W, GRID_H)
                
                last_speed_var = speed_var
                last_dir_entropy = dir_entropy
                last_avg_speed = avg_speed
                last_direction_coherence = direction_coherence
                last_speed_magnitude = speed_magnitude
            elif prev_gray is not None:
                speed_var = last_speed_var
                dir_entropy = last_dir_entropy
                avg_speed = last_avg_speed
                direction_coherence = last_direction_coherence
                speed_magnitude = last_speed_magnitude
            else:
                speed_var = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                dir_entropy = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                avg_speed = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                direction_coherence = np.zeros((GRID_H, GRID_W), dtype=np.float32)
                speed_magnitude = np.zeros((GRID_H, GRID_W), dtype=np.float32)
            
            prev_gray = gray

            # Risk calculations
            pressure = density * speed_var
            Dn = np.clip(minmax01(density) * 6.0, 0, 1)
            Fn = np.clip(minmax01(0.5 * dir_entropy + 0.5 * minmax01(speed_var)) * 4.0, 0, 1)
            Pn = np.clip(minmax01(pressure) * 8.0, 0, 1)

            panic_pressure = density * avg_speed
            coherent_risk = density * direction_coherence * speed_magnitude

            PPn = np.clip(minmax01(panic_pressure) * 10.0, 0, 1)
            CRn = np.clip(minmax01(coherent_risk) * 12.0, 0, 1)
            ASn = np.clip(minmax01(avg_speed) * 8.0, 0, 1)
            DCn = np.clip(minmax01(direction_coherence) * 6.0, 0, 1)

            W_D, W_F, W_P = 0.10, 0.06, 0.08
            W_PP, W_CR, W_AS, W_DC = 0.14, 0.12, 0.08, 0.05
            W_C, W_CH = 0.25, 0.12

            R_original = W_D * Dn + W_F * Fn + W_P * Pn
            R_panic = W_PP * PPn + W_CR * CRn + W_AS * ASn + W_DC * DCn
            R_detection = W_C * count_risk + W_CH * change_risk

            R_cell = R_original + R_panic + R_detection
            R_global = float(R_cell.mean())

            # Continuous baseline learning
            baseline.add_sample(R_global, float(density.mean()), float(avg_speed.mean()), count)

            if baseline.should_update_baseline():
                logger.console_log(f"🔄 Updating baseline (Frame {baseline.get_total_frames()})...", "LEARNING")
                try:
                    update_success = baseline.update_baseline()
                    if update_success:
                        TH_LOW_ADAPTIVE, TH_HIGH_ADAPTIVE = baseline.get_thresholds()
                except Exception as e:
                    logger.console_log(f"⚠ Baseline update error: {e}", "WARNING")

            # PER-CAMERA baseline mode control
            total_frames = baseline.get_total_frames()
            can_use = baseline.can_use_baseline_mode()

            # Get this camera's baseline settings
            cam_baseline_dict = baseline_mode_dict[camera_id]
            
            if not can_use:
                baseline_mode_enabled = False
                if cam_baseline_dict.get('enabled', False):
                    temp = dict(cam_baseline_dict)
                    temp['enabled'] = False
                    cam_baseline_dict.clear()
                    cam_baseline_dict.update(temp)
            else:
                baseline_mode_enabled = cam_baseline_dict.get('enabled', True)
                if not cam_baseline_dict.get('auto_enabled', False):
                    temp = dict(cam_baseline_dict)
                    temp['enabled'] = True
                    temp['auto_enabled'] = True
                    cam_baseline_dict.clear()
                    cam_baseline_dict.update(temp)
                    logger.console_log("✅ 5000 frames reached! Baseline mode AUTO-ENABLED", "SUCCESS")

            ema_cell = ema_update(ema_cell, R_original + R_panic, ALPHA)
            prev_global = R_global if prev_global is None else prev_global
            
            if baseline_mode_enabled and baseline.is_calibrated():
                deviation_risk = baseline.get_deviation_risk(R_global)
                R_global_final = BASE_WEIGHT * R_global + DEVIATION_WEIGHT * deviation_risk
                S_global = ALPHA * R_global_final + (1 - ALPHA) * prev_global
                TH_LOW = TH_LOW_ADAPTIVE
                TH_HIGH = TH_HIGH_ADAPTIVE
            else:
                deviation_risk = 0.0
                S_global = ALPHA * R_global + (1 - ALPHA) * prev_global
                TH_LOW = TH_LOW_FIXED
                TH_HIGH = TH_HIGH_FIXED

            R_buffer.append(S_global)
            if len(R_buffer) == R_buffer.maxlen:
                S_global_median = np.median(R_buffer)
            else:
                S_global_median = S_global

            deviation_score = baseline.get_deviation_score(S_global_median)

            # OPTIMIZED: STEP adjustment (never drops below 2)
            if S_global_median < TH_LOW:
                STEP = min(MAX_STEP, max(STEP + 1, 3))
            elif S_global_median > TH_HIGH:
                STEP = max(MIN_STEP, STEP - 1)

            if alert_state != "ALERT" and S_global_median >= TH_HIGH:
                alert_state = "ALERT"
                alert_start_time = time.time()
                
                mode_str = "Baseline ON" if baseline_mode_enabled else "Baseline OFF"
                logger.console_log(f"🚨 ALERT! [{mode_str}] Risk={S_global_median:.3f}, Base={R_global:.3f}, Det={count}", "ALERT")

                trigger_reason = []
                if count > MAX_DETECTIONS * 0.7:
                    trigger_reason.append(f"High_Count({count})")
                if S_global_median > TH_HIGH:
                    trigger_reason.append(f"High_Risk({S_global_median:.3f})")
                if change_risk > 0.5:
                    trigger_reason.append(f"Sudden_Change({change_risk:.3f})")
                if baseline_mode_enabled and abs(deviation_score) > 2.5:
                    trigger_reason.append(f"Anomaly({deviation_score:.2f}σ)")

                logger.log_alert({
                    'alert_type': 'RISK_ALERT',
                    'risk_score': S_global_median,
                    'base_risk': R_global,
                    'detections': count,
                    'trigger_reason': ', '.join(trigger_reason) or 'Threshold_Exceeded',
                    'duration': 0,
                    'deviation_score': deviation_score,
                    'baseline_mode': baseline_mode_enabled
                })

            elif alert_state != "OK" and S_global_median <= TH_LOW:
                alert_state = "OK"
                if alert_start_time:
                    duration = time.time() - alert_start_time
                    logger.console_log(f"✅ Alert cleared after {duration:.1f}s", "SUCCESS")
                    alert_start_time = None

            # Performance metrics
            frame_end_time = time.time()
            processing_time = (frame_end_time - frame_start_time) * 1000
            current_fps = 1.0 / (frame_end_time - last_frame_time) if last_frame_time else 0
            last_frame_time = frame_end_time

            # Log frame data
            if frame_idx % FRAME_LOG_INTERVAL == 0:
                logger.log_frame({
                    'frame_num': frame_idx,
                    'base_risk': R_global,
                    'risk_score': S_global_median,
                    'alert_status': alert_state,
                    'baseline_mode': baseline_mode_enabled,
                    'detections': count,
                    'chaos_risk': R_original.mean(),
                    'panic_risk': R_panic.mean(),
                    'change_risk': change_risk,
                    'deviation_risk': deviation_risk,
                    'fps': current_fps,
                    'processing_time': processing_time,
                    'deviation_score': deviation_score,
                    'th_low': TH_LOW,
                    'th_high': TH_HIGH
                })

                mode_str = "[OPT]" + ("[B:ON]" if baseline_mode_enabled else "[B:OFF]")
                status = f"{mode_str} Frame:{frame_idx} | Risk:{S_global_median:.3f} | FPS:{current_fps:.1f}"
                log_level = "ALERT" if alert_state == "ALERT" else "INFO"
                logger.console_log(status, log_level)

            # Visualization
            vis = frame.copy()

            if ema_cell is not None:
                cell_w, cell_h = grid_bins(W, H, GRID_W, GRID_H)
                for i in range(GRID_H):
                    for j in range(GRID_W):
                        v = ema_cell[i, j]
                        x0 = int(j * cell_w)
                        y0 = int(i * cell_h)
                        x1 = int(min((j + 1) * cell_w, W))
                        y1 = int(min((i + 1) * cell_h, H))

                        overlay = vis[y0:y1, x0:x1].copy()
                        color = color_map(v)
                        cv2.rectangle(overlay, (0, 0), (x1-x0, y1-y0), color, -1)
                        vis[y0:y1, x0:x1] = cv2.addWeighted(vis[y0:y1, x0:x1], 0.7, overlay, 0.3, 0)
                        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)

            mode_indicator = "[OPT]" + ("[B:ON]" if baseline_mode_enabled else "[B:OFF]")
            txt = f"{camera_id} {mode_indicator} | Risk: {S_global_median:.3f} | {alert_state} | Det: {count}"
            color = (0, 0, 255) if alert_state == "ALERT" else (0, 255, 0)

            cv2.rectangle(vis, (5, 5), (W-5, 40), (0, 0, 0), -1)
            cv2.putText(vis, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            # Update shared data
            shared_dict[camera_id] = {
                'risk_score': float(S_global_median),
                'base_risk': float(R_global),
                'alert': alert_state == "ALERT",
                'count': int(count),
                'timestamp': time.time(),
                'chaos_risk': float(R_original.mean()),
                'panic_risk': float(R_panic.mean()),
                'change_risk': float(change_risk),
                'deviation_risk': float(deviation_risk),
                'fps': float(current_fps),
                'frame_num': int(frame_idx),
                'deviation_score': float(deviation_score),
                'th_low': float(TH_LOW),
                'th_high': float(TH_HIGH),
                'calibrated': baseline.is_calibrated(),
                'baseline_mode': baseline_mode_enabled,
                'total_baseline_frames': total_frames,
                'can_use_baseline': can_use
            }

            # Send frame to GUI
            try:
                frame_queue.put_nowait((camera_id, vis))
            except:
                pass

            # Send log messages
            try:
                log_queue.put_nowait({
                    'camera_id': camera_id,
                    'timestamp': time.strftime("%H:%M:%S"),
                    'message': f"{status}",
                    'level': log_level
                })
            except:
                pass

            frame_idx += 1

        video_reader.stop()
        logger.console_log("Completed", "SUCCESS")

    except Exception as e:
        print(f"[{camera_id}] ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
