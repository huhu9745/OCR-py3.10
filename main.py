"""
荒料 OCR 四图一致性检测服务
----------------------------------
功能流程：
1. 接收4张荒料图片
2. 分别进行OCR识别
3. 打印识别到的所有文本
4. 提取编号数字
5. 判断4个结果是否一致
6. 一致返回编号
7. 不一致返回错误
"""

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import re
import uvicorn
import traceback
from ocr_detect import OCRRec



# ---------------OCR 服务----------------

class OCRService:
    """
    负责：
    1. 加载OCR模型
    2. 读取图片
    3. OCR识别
    4. 编号提取
    5. 一致性判断
    """

    def __init__(self, device="gpu:0"):
        """
        初始化OCR模型

        device:
            "gpu:0" 使用GPU
            "cpu"   使用CPU
        """
        self.model = OCRRec(device=device)


    # -------------上传图片转OpenCV格式-----------------------

    def read_image(self, file: UploadFile):
        data = file.file.read()
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img


    #---------- 从OCR文本中提取编号数字--------------

    def extract_number(self, texts):
        """
        从识别文本中提取第一个数字串
        示例：
        ["BLK", "20250318"] -> 20250318
        """
        for t in texts:
            match = re.search(r"\d+", t)
            if match:
                return match.group()
        return None


    # ------------OCR读图识别-------------------------

    def process(self, files):
        results = []

        for idx, file in enumerate(files):
            print(f"\n========== 第 {idx + 1} 张荒料图片 ==========")

            img = self.read_image(file)

            if img is None:
                return False, {
                    "error": f"第{idx + 1}张图片读取失败"
                }

            texts = self.model.predict(img)

            if not texts:
                return False, {
                    "error": f"第{idx + 1}张图片未识别到文本"
                }

            print("识别到的文本内容：")

            for i, t in enumerate(texts):
                print(f"  文本{i + 1}: {t}")

            number = self.extract_number(texts)

            print(f"提取出的荒料编号: {number}")

            if number is None:
                return False, {
                    "error": f"第{idx + 1}张图片未识别到编号"
                }

            results.append(number)

        print("\n========== 四张荒料图片最终结果 ==========")
        print("所有识别编号:", results)

        if len(set(results)) == 1:
            return True, {
                "value": results[0],
                "all_results": results
            }

        return False, {
            "error": "四张图片识别结果不一致",
            "all_results": results
        }


#-------------------- FastAPI 接口------------------

app = FastAPI()

# 创建 OCR 服务实例
# GPU不稳定可改为 device="cpu"
ocr_service = OCRService(device="gpu:0")


# ---------------------------------
# 首页接口--测试用
# ---------------------------------
@app.get("/")
def root():
    return {"msg": "Stone OCR server running"}


# ---------------------------------
# OCR 接口
# ---------------------------------
@app.post("/ocr")
async def ocr_api(
        file1: UploadFile = File(...),
        file2: UploadFile = File(...),
        file3: UploadFile = File(...),
        file4: UploadFile = File(...)
):
    try:
        files = [file1, file2, file3, file4]

        success, result = ocr_service.process(files)

        if success:
            return {
                "success": True,
                "value": result["value"],
                "all_results": result["all_results"]
            }

        return JSONResponse(
            {"success": False, **result},
            status_code=400
        )

    except Exception as e:
        print("========== 接口异常 ==========")
        traceback.print_exc()
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )



# -------------启动服务-----------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )