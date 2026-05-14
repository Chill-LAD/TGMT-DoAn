"""
Wrapper module để tương thích ngược với code cũ.
Khuyến nghị sử dụng model_v1.py và model_v2.py trực tiếp.
"""
from model_v2 import create_model_v2 as create_model
from model_v2 import HybridWatermarkModelV2 as HybridWatermarkModel
from model_v2 import WatermarkDetectorV2 as WatermarkDetector

__all__ = ['create_model', 'HybridWatermarkModel', 'WatermarkDetector']