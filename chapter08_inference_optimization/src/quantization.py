from typing import Dict, Tuple
from loguru import logger
import numpy as np


class QuantizationAlgorithms:

    @staticmethod
    def absmax_quantize(weights: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Cuantización Absmax Simétrica (INT8):
        scale = 127 / max(|X|)
        X_quant = round(scale * X)
        """
        max_abs = float(np.max(np.abs(weights)))
        scale = 127.0 / max(max_abs, 1e-8)
        quantized = np.clip(np.round(scale * weights), -127, 127).astype(np.int8)
        return quantized, scale

    @staticmethod
    def absmax_dequantize(quantized: np.ndarray, scale: float) -> np.ndarray:
        return quantized.astype(np.float32) / scale

    @staticmethod
    def zeropoint_quantize(weights: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        Cuantización Asimétrica Zero-Point (INT8):
        scale = 255 / (max(X) - min(X))
        zeropoint = -round(scale * min(X)) - 128
        """
        min_val = float(np.min(weights))
        max_val = float(np.max(weights))
        val_range = max(max_val - min_val, 1e-8)

        scale = 255.0 / val_range
        zeropoint = -np.round(scale * min_val) - 128.0
        quantized = np.clip(np.round(scale * weights + zeropoint), -128, 127).astype(np.int8)
        return quantized, scale, zeropoint

    @staticmethod
    def zeropoint_dequantize(quantized: np.ndarray, scale: float, zeropoint: float) -> np.ndarray:
        return (quantized.astype(np.float32) - zeropoint) / scale

    @classmethod
    def evaluate_quantization_error(cls, num_params: int = 100_000) -> Dict[str, float]:
        np.random.seed(42)
        # Distribución normal con presencia de outliers en el 0.1% de los pesos
        weights = np.random.normal(loc=0.0, scale=0.5, size=num_params).astype(np.float32)
        outlier_indices = np.random.choice(num_params, size=int(num_params * 0.001), replace=False)
        weights[outlier_indices] *= 8.0

        # 1. Absmax
        q_abs, scale_abs = cls.absmax_quantize(weights)
        deq_abs = cls.absmax_dequantize(q_abs, scale_abs)
        mse_abs = float(np.mean((weights - deq_abs) ** 2))

        # 2. Zero-point
        q_zp, scale_zp, zp = cls.zeropoint_quantize(weights)
        deq_zp = cls.zeropoint_dequantize(q_zp, scale_zp, zp)
        mse_zp = float(np.mean((weights - deq_zp) ** 2))

        # Ahorro de memoria teórico: FP16 (2 bytes) a INT8 (1 byte) -> 50% reducción
        memory_reduction = 50.0

        logger.info(f"Quantization Evaluation -> Absmax MSE: {mse_abs:.6f} | Zero-Point MSE: {mse_zp:.6f}")
        return {
            "absmax_mse": round(mse_abs, 6),
            "zeropoint_mse": round(mse_zp, 6),
            "memory_reduction_pct": memory_reduction,
        }
