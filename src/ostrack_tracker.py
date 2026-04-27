"""
📌 OSTrack Boss Engine — The Apex Self-Aware Edition (V4 - Physics Locked)
✅ Base: Stable Working Code (GPU/CPU Safe)
✅ Architecture: State Machine + Physics Locks (Min/Max Size, Strict Trajectory, Expansion Cap)
"""
import sys
import torch
import cv2
import math
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# ==========================================
# 🛠️ Legacy Compatibility & CPU Patch 
# ==========================================
if 'torch._six' not in sys.modules:
    class _Six:
        string_classes = (str,)
        int_classes = (int,)
    sys.modules['torch._six'] = _Six()
    torch._six = _Six()

if 'visdom' not in sys.modules:
    mock_visdom = MagicMock()
    sys.modules['visdom'] = mock_visdom
    sys.modules['visdom.server'] = mock_visdom

# ==========================================
# 🛑 PATH HACK FOR OSTRACK
# ==========================================
if "./OSTrack" not in sys.path:
    sys.path.insert(0, "./OSTrack")

from lib.test.tracker.ostrack import OSTrack
from lib.test.parameter.ostrack import parameters

CONFIG_NAME = "vitb_256_mae_ce_32x4_ep300"
WEIGHTS_PATH = "./models/OSTrack_ep0300.pth.tar" 

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⏳ Loading OSTrack (ViT-Base) on {device}...")

def build_ostrack_params():
    params = parameters(CONFIG_NAME)
    params.checkpoint = WEIGHTS_PATH 
    params.debug = False 
    params.save_all_boxes = False 
    return params

ostrack_params = build_ostrack_params()
ostrack_model = OSTrack(ostrack_params, "ostrack")

def clip_box(box, w, h):
    return [
        max(0.0, min(box[0], w - 1)),
        max(0.0, min(box[1], h - 1)),
        max(2.0, min(box[2], w - box[0])),
        max(2.0, min(box[3], h - box[1]))
    ]

def run_tracker(sequence, visualize=False):
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]
    boxes = sequence.get("boxes", None)
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    h_img, w_img = frame.shape[:2]
    x, y, w, h = map(float, init_bbox)
    init_w, init_h = w, h  # 🔒 1. حفظ الحجم الأصلي لحماية الـ Micro-Targets
    
    init_info = {'init_bbox': [x, y, w, h]}
    ostrack_model.initialize(frame, init_info)
    
    csv_box = [x, y, w, h]
    frame_idx = 1
    
    # 🧠 STATE MACHINE VARIABLES
    vx, vy = 0.0, 0.0
    smooth_score = 1.0
    lost_counter = 0
    state = "TRACKING"
    
    # 🧠 Temporal Consistency History
    score_history = []
    
    predictions = [{
        "id": f"{seq_name}_0", 
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)
    }]

    if visualize:
        cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            outputs = ostrack_model.track(frame)
            raw_score = float(outputs.get('conf_score', outputs.get('best_score', 1.0)))
            
            raw_x, raw_y, raw_w, raw_h = map(float, outputs['target_bbox'])

            if raw_w <= 2 or raw_h <= 2 or np.isnan(np.array([raw_x, raw_y, raw_w, raw_h])).any():
                raw_x, raw_y, raw_w, raw_h = csv_box
                raw_score = 0.0

            # 🌊 1. Score History & Stability
            score_history.append(raw_score)
            if len(score_history) > 5:
                score_history.pop(0)
            
            stable_score = sum(score_history) / len(score_history) if score_history else raw_score
            
            if frame_idx == 1:
                smooth_score = raw_score
            else:
                smooth_score = 0.8 * smooth_score + 0.2 * raw_score
                
            effective_score = min(stable_score, smooth_score)

            # 🤖 2. Adaptive Thresholds & State Definition
            track_th = 0.5 + 0.2 * effective_score
            uncertain_th = 0.3 + 0.1 * effective_score

            if effective_score > track_th:
                state = "TRACKING"
                lost_counter = 0
            elif effective_score > uncertain_th:
                if state != "LOST":
                    state = "UNCERTAIN"
            else:
                state = "LOST"

            pot_vx, pot_vy = 0.0, 0.0

            if frame_idx > 1:
                raw_cx = raw_x + raw_w / 2.0
                raw_cy = raw_y + raw_h / 2.0
                prev_cx = csv_box[0] + csv_box[2] / 2.0
                prev_cy = csv_box[1] + csv_box[3] / 2.0
                
                pot_vx = raw_cx - prev_cx
                pot_vy = raw_cy - prev_cy
                
                jump = math.hypot(pot_vx, pot_vy)
                diag = math.hypot(csv_box[2], csv_box[3])
                current_speed = math.hypot(vx, vy)
                center_drift = jump / (diag + 1e-6)

                # 👁️ Object Awareness
                is_small = (csv_box[2] * csv_box[3]) < 1500

                # 🚀 Dynamic Drift Limit
                drift_limit = 4.0 if current_speed > diag else 2.5
                if is_small:
                    drift_limit *= 1.5

                # 🎯 Adaptive Jump Gating + Progressive Expansion
                allowed_jump = 1.5 * diag
                if current_speed > (0.8 * diag) or effective_score > 0.75:
                    allowed_jump = 3.0 * diag
                elif effective_score < 0.60:
                    allowed_jump = 1.0 * diag
                
                if is_small:
                    allowed_jump *= 1.5
                    
                # 🔒 3. Search Expansion when LOST (Capped)
                if state == "LOST":
                    allowed_jump = min(allowed_jump * (1 + lost_counter * 0.1), 5.0 * diag)

                # 🛡️ VALIDATION (Area, Scale, & Direction Penalty)
                area_ratio = (raw_w * raw_h) / (csv_box[2] * csv_box[3])
                scale_change_w = abs(raw_w - csv_box[2]) / (csv_box[2] + 1e-6)
                scale_change_h = abs(raw_h - csv_box[3]) / (csv_box[3] + 1e-6)
                
                direction_change = 0.0
                if current_speed > 2.0 and jump > 2.0: 
                    direction_change = abs(math.atan2(vy, vx) - math.atan2(pot_vy, pot_vx))

                # نظام الغرامات الذكي
                penalty = 0.0
                if jump > allowed_jump:
                    penalty += 0.4
                if center_drift > drift_limit:
                    penalty += 0.3
                if direction_change > (math.pi / 2):
                    penalty += 0.4  # 🔒 2. غرامة مضاعفة لكسر الاتجاه الحاد لحل التوائم
                if not (0.4 < area_ratio < 2.5) or scale_change_w > 0.5 or scale_change_h > 0.5:
                    penalty += 0.3 
                
                effective_score *= max(0.0, 1.0 - penalty)

                if state != "LOST":
                    if effective_score > track_th:
                        state = "TRACKING"
                    elif effective_score > uncertain_th:
                        state = "UNCERTAIN"
                    else:
                        state = "LOST"

            # ==========================================
            # 🔄 STATE MACHINE EXECUTION
            # ==========================================
            if frame_idx > 1:
                if state in ["TRACKING", "UNCERTAIN"]:
                    lost_counter = 0 
                    current_alpha = 0.85 * effective_score
                    
                    # Update Position
                    csv_box[0] = current_alpha * raw_x + (1 - current_alpha) * csv_box[0]
                    csv_box[1] = current_alpha * raw_y + (1 - current_alpha) * csv_box[1]

                    # Update Scale with Motion Check
                    if state == "TRACKING":
                        scale_alpha = min(0.4, (0.2 * effective_score) if is_small else (0.5 * effective_score))
                        if (abs(vx) + abs(vy)) > diag and effective_score < 0.70:
                            scale_alpha *= 0.5
                    else: 
                        scale_alpha = 0.05 * effective_score
                        
                    csv_box[2] = scale_alpha * raw_w + (1 - scale_alpha) * csv_box[2]
                    csv_box[3] = scale_alpha * raw_h + (1 - scale_alpha) * csv_box[3]

                    # 🔒 1. Min/Max Size Lock (حماية الأهداف من التلاشي أو الابتلاع)
                    csv_box[2] = max(8.0, min(csv_box[2], init_w * 4.0))
                    csv_box[3] = max(8.0, min(csv_box[3], init_h * 4.0))

                    # 🟢 Safe Velocity Learning
                    if effective_score > 0.60 and center_drift < 2.0:
                        vx = 0.7 * vx + 0.3 * pot_vx
                        vy = 0.7 * vy + 0.3 * pot_vy

                elif state == "LOST":
                    lost_counter += 1
                    
                    if raw_score > 0.50 and center_drift < 3.0:
                        state = "UNCERTAIN"
                        lost_counter = 0
                        csv_box[0] = 0.3 * raw_x + 0.7 * csv_box[0]
                        csv_box[1] = 0.3 * raw_y + 0.7 * csv_box[1]
                    else:
                        if lost_counter < 15: 
                            max_ghost_move = 0.5 * math.hypot(csv_box[2], csv_box[3])
                            move = math.hypot(vx, vy)
                            
                            if move > max_ghost_move:
                                scale = max_ghost_move / (move + 1e-6)
                                vx *= scale
                                vy *= scale
                                
                            csv_box[0] += vx * 0.5
                            csv_box[1] += vy * 0.5
                            
                            vx *= 0.85
                            vy *= 0.85
                            if abs(vx) < 0.5: vx = 0.0
                            if abs(vy) < 0.5: vy = 0.0
                        else:
                            vx, vy = 0.0, 0.0

            else:
                csv_box = [raw_x, raw_y, raw_w, raw_h]

            csv_box = clip_box(csv_box, w_img, h_img)
            
            predictions.append({
                "id": f"{seq_name}_{frame_idx}",
                "x": round(csv_box[0], 2),
                "y": round(csv_box[1], 2),
                "w": round(csv_box[2], 2),
                "h": round(csv_box[3], 2)
            })

            # ==========================================
            # 🎬 VISUALIZATION
            # ==========================================
            if visualize:
                color_map = {"TRACKING": (0, 255, 0), "UNCERTAIN": (0, 165, 255), "LOST": (0, 0, 255)}
                color = color_map[state]

                vx_vis, vy_vis, vw_vis, vh_vis = map(int, csv_box)
                cv2.rectangle(frame, (vx_vis, vy_vis), (vx_vis + vw_vis, vy_vis + vh_vis), color, 2)

                cv2.putText(frame, f"State: {state}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Eff Score: {effective_score:.2f} | Raw: {raw_score:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_idx} | Lost: {lost_counter}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(frame, f"Seq: {seq_name}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                if boxes is not None and frame_idx < len(boxes):
                    xg, yg, wg, hg = map(int, boxes[frame_idx])
                    if wg > 0 and hg > 0:
                        cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

                cv2.imshow("Tracking (Prediction)", frame)
                key = cv2.waitKey(10) & 0xFF
                if key == 27: break
                if key == ord('q'): return {"status": "stop", "predictions": predictions}

            frame_idx += 1

    cap.release()
    if visualize: cv2.destroyAllWindows()
    return {"status": "done", "predictions": predictions}