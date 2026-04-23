"""
📌 OSTrack Boss Engine — The One-Stream Barrier Breaker
✅ Model: ViT-Base 256 (One-Stream Transformer)
✅ Features: SiamRPN-Style Visualization & Kaggle 'public_id' Submission Export
"""
import sys
import torch
import cv2
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

# ==========================================
# 🛠️ Legacy Compatibility & CPU Patch (Ninja Hack v3)
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

# Force CPU Bypass (شغالين بيه حالياً للتيست)
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
WEIGHTS_PATH = "./models/ostrack_ep0300.pth.tar" 

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
    boxes = sequence.get("boxes", None) # لو في Ground Truth هيترسم
    
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
    
    # 👈 تعديل الـ ID ليكون public_id عشان كاجل
    predictions = [{
        "public_id": f"{seq_name}_0", 
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)
    }]

    # 🎬 تضبيط الشاشة زي ستايل SiamRPN لسمير
    if visualize:
        cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = ostrack_model.track(frame)
        raw_box = outputs['target_bbox']
        score = float(outputs.get('conf_score', outputs.get('best_score', 1.0)))
        
        raw_x, raw_y, raw_w, raw_h = raw_box

        # حالة التتبع والـ Alpha للـ Visualization
        tracking_ok = score > 0.50
        current_alpha = 0.0

        if raw_w <= 2 or raw_h <= 2 or np.isnan(np.array([raw_x, raw_y, raw_w, raw_h])).any():
            raw_x, raw_y, raw_w, raw_h = csv_box
            score = 0.0
            tracking_ok = False

        if frame_idx > 1:
            if tracking_ok:
                current_alpha = 0.85
                csv_box[0] = current_alpha * raw_x + (1 - current_alpha) * csv_box[0]
                csv_box[1] = current_alpha * raw_y + (1 - current_alpha) * csv_box[1]
                csv_box[2] = current_alpha * raw_w + (1 - current_alpha) * csv_box[2]
                csv_box[3] = current_alpha * raw_h + (1 - current_alpha) * csv_box[3]
            else:
                decay = 0.95
                csv_box[0] = decay * csv_box[0] + (1 - decay) * raw_x
                csv_box[1] = decay * csv_box[1] + (1 - decay) * raw_y
        else:
            csv_box = [raw_x, raw_y, raw_w, raw_h]

        csv_box = clip_box(csv_box, w_img, h_img)
        
        # 👈 حفظ بـ public_id
        predictions.append({
            "public_id": f"{seq_name}_{frame_idx}",
            "x": round(csv_box[0], 2),
            "y": round(csv_box[1], 2),
            "w": round(csv_box[2], 2),
            "h": round(csv_box[3], 2)
        })

        # ==========================================
        # 🎬 VISUALIZATION (SIAMRPN STYLE)
        # ==========================================
        if visualize:
            color = (255, 0, 0) if tracking_ok else (0, 0, 255) # أزرق تتبع، أحمر ضياع
            label = "Tracking" if tracking_ok else "Frozen (Soft)"

            vx, vy, vw, vh = map(int, csv_box)
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), color, 2)

            # رصة الكلام بالمللي زي ما طلبتم
            cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Score: {score:.2f} | Alpha: {current_alpha:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Seq: {seq_name}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # رسم الـ GT لو موجود
            if boxes is not None and frame_idx < len(boxes):
                xg, yg, wg, hg = map(int, boxes[frame_idx])
                if wg > 0 and hg > 0:
                    cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

            cv2.imshow("Tracking (Prediction)", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27: # ESC
                break
            if key == ord('q'):
                return {"status": "stop", "predictions": predictions}

        frame_idx += 1

    cap.release()
    if visualize:
        cv2.destroyAllWindows()
    return {"status": "done", "predictions": predictions}