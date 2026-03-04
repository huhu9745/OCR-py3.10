# 验证安装是否成功
import paddleocr


print(f"PaddleOCR版本: {paddleocr.__version__}")

# 验证GPU是否可用
import paddle
print(f"Paddle版本: {paddle.__version__}")
print(f"GPU可用: {paddle.is_compiled_with_cuda()}")
print(f"GPU数量: {paddle.device.cuda.device_count()}")