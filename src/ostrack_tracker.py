"""
📌 OSTrack Boss Engine — The Adaptive Edition
✅ Base: Stable Working Code
✅ Logic: Dynamic Alpha, Smart Ghost Tracking, and Adaptive Jump Gating
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

# Force CPU Bypass
torch.nn.Module.cuda = lambda self, *args, **kwargs: self.cpu()
torch.Tensor.cuda = lambda self, *args, **kwargs: self.cpu()

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

def run_tracker(sequence, visualize=True):
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
    
    init_info = {'init_bbox': [x, y, w, h]}
    ostrack_model.initialize(frame, init_info)
    
    csv_box = [x, y, w, h]
    frame_idx = 1
    
    # 🧠 Variables for Smart Logic
    vx, vy = 0.0, 0.0
    
    predictions = [{
        "public_id": f"{seq_name}_0", 
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
            score = float(outputs.get('conf_score', outputs.get('best_score', 1.0)))
            
            raw_x, raw_y, raw_w, raw_h = map(float, outputs['target_bbox'])

            if raw_w <= 2 or raw_h <= 2 or np.isnan(np.array([raw_x, raw_y, raw_w, raw_h])).any():
                raw_x, raw_y, raw_w, raw_h = csv_box
                score = 0.0

            tracking_ok = False
            pot_vx, pot_vy = 0.0, 0.0
            current_alpha = 0.0

            if frame_idx > 1 and score > 0.45:
                raw_cx = raw_x + raw_w / 2.0
                raw_cy = raw_y + raw_h / 2.0
                prev_cx = csv_box[0] + csv_box[2] / 2.0
                prev_cy = csv_box[1] + csv_box[3] / 2.0
                
                pot_vx = raw_cx - prev_cx
                pot_vy = raw_cy - prev_cy
                
                jump = math.hypot(pot_vx, pot_vy)
                diag = math.hypot(csv_box[2], csv_box[3])
                current_speed = math.hypot(vx, vy)

                # ==========================================
                # 🎯 1. ADAPTIVE JUMP GATING (Simulated Dynamic Search)
                # ==========================================
                # Default Base (يحمي من المشتتات العادية)
                allowed_jump = 1.5 * diag

                # 🟠 Fast Motion Case (UAVs / Birds)
                # لو الهدف كان بيجري بسرعة من الفريم اللي فات، نفتحله المساحة
                if current_speed > (0.8 * diag) or score > 0.75:
                    allowed_jump = 3.0 * diag
                
                # 🔴 Distractor Case (Motorcycle / Groups)
                # لو السكور نص نص ومفيش سرعة عالية سابقة، نقفل المساحة جداً لمنع قفزات الهوية
                elif score < 0.60:
                    allowed_jump = 1.0 * diag

                # ==========================================
                # 🛡️ VALIDATION
                # ==========================================
                if jump <= allowed_jump:
                    # Size Check (يمنع الانفجار المفاجئ للبوكس)
                    size_ratio_w = raw_w / csv_box[2]
                    size_ratio_h = raw_h / csv_box[3]
                    if (0.55 < size_ratio_w < 1.6) and (0.55 < size_ratio_h < 1.6):
                        tracking_ok = True

            # ==========================================
            # 🔄 UPDATE LOGIC
            # ==========================================
            if frame_idx > 1:
                if tracking_ok:
                    # 🔵 Dynamic Alpha (فكرتك العبقرية)
                    current_alpha = 0.85 * score
                    
                    csv_box[0] = current_alpha * raw_x + (1 - current_alpha) * csv_box[0]
                    csv_box[1] = current_alpha * raw_y + (1 - current_alpha) * csv_box[1]
                    csv_box[2] = current_alpha * raw_w + (1 - current_alpha) * csv_box[2]
                    csv_box[3] = current_alpha * raw_h + (1 - current_alpha) * csv_box[3]

                    # تحديث السرعة في حالة الثقة فقط
                    if score > 0.60:
                        vx, vy = pot_vx, pot_vy
                else:
                    # 🟡 Smart Ghost Tracking (Occlusion Fix)
                    # السرعة بتضرب في 0.5 عشان الموديل ميسرحش بسرعة
                    csv_box[0] += vx * 0.5
                    csv_box[1] += vy * 0.5
                    
                    vx *= 0.85
                    vy *= 0.85
                    
                    if abs(vx) < 0.5: vx = 0.0
                    if abs(vy) < 0.5: vy = 0.0
            else:
                csv_box = [raw_x, raw_y, raw_w, raw_h]

            csv_box = clip_box(csv_box, w_img, h_img)
            
            predictions.append({
                "public_id": f"{seq_name}_{frame_idx}",
                "x": round(csv_box[0], 2),
                "y": round(csv_box[1], 2),
                "w": round(csv_box[2], 2),
                "h": round(csv_box[3], 2)
            })

            # ==========================================
            # 🎬 VISUALIZATION
            # ==========================================
            if visualize:
                color = (255, 0, 0) if tracking_ok else (0, 0, 255) 
                label = "Tracking" if tracking_ok else "Adaptive Ghost"

                vx_vis, vy_vis, vw_vis, vh_vis = map(int, csv_box)
                cv2.rectangle(frame, (vx_vis, vy_vis), (vx_vis + vw_vis, vy_vis + vh_vis), color, 2)

                cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Score: {score:.2f} | Alpha: {current_alpha:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_idx}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
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