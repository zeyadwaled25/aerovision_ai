import pandas as pd

def run_simple_tracker(sequence):
    import cv2

    frames = sequence["frames"]
    seq_name = sequence["seq_name"]

    results = []

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # جرّب threshold أقل
        _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
        else:
            x, y, w, h = 0, 0, 0, 0

        print(f"Frame {i}: {x}, {y}")  # debug

        results.append({
            "id": f"{seq_name}_{i}",
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

    return results

def visualize_tracking(frames, results):
    import cv2

    for i, frame in enumerate(frames):
        x = int(results[i]["x"])
        y = int(results[i]["y"])
        w = int(results[i]["w"])
        h = int(results[i]["h"])

        # رسم البوكس
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Tracking", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

def save_results(all_results, path):
    df = pd.DataFrame(all_results)
    df.to_csv(path, index=False)