# ocr_model.py
import os
import cv2
import numpy as np
import warnings
import time

warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from paddleocr import TextDetection, TextRecognition


class OCRRec:

    def __init__(self, device="gpu:0"):
        self.det_model = TextDetection(
            model_dir=r"D:\conch\OCR-py3.10\OCR\PP-OCRv5_server_det_infer\PP-OCRv5_server_det_infer",
            device=device
        )

        self.rec_model = TextRecognition(
            model_dir=r"D:\conch\OCR-py3.10\OCR\PP-OCRv5_server_rec_infer\PP-OCRv5_server_rec_infer",
            device=device
        )

    def predict(self, img):
        start_time = time.time()

        results = []

        det_out = self.det_model.predict(img)

        for det in det_out:
            polys = det["dt_polys"]

            for poly in polys:
                pts = poly.astype(np.int32)

                x_min = np.min(pts[:, 0])
                x_max = np.max(pts[:, 0])
                y_min = np.min(pts[:, 1])
                y_max = np.max(pts[:, 1])

                crop = img[y_min:y_max, x_min:x_max]

                rec_out = self.rec_model.predict(crop)

                for rec in rec_out:
                    results.append(rec["rec_text"])

        print("模型推理时间", round(time.time() - start_time, 4))
        return results