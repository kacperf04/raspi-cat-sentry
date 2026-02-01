from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "cat_detector_1.onnx"

SESSION = ort.InferenceSession(str(MODEL_PATH))

def preprocess(image):
    """Normalizes and rezises the input image to match the model's input size."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        orig_h, orig_w = frame.shape[:2]
        input_tensor = preprocess(frame)

        outputs = SESSION.run(None, {"images": input_tensor})
        detections = outputs[0]
        detections = np.squeeze(detections)

        for i in range(len(detections)):
            x1, y1, x2, y2, score, class_id = detections[i]

            if score < 0.4:
                continue

            x1 = int(x1 / 640 * orig_w)
            y1 = int(y1 / 640 * orig_h)
            x2 = int(x2 / 640 * orig_w)
            y2 = int(y2 / 640 * orig_h)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
            label = f"Cat: {score:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5, (255, 0, 0),
                3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()