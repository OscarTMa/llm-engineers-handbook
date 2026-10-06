from typing import List, Tuple
from loguru import logger
import numpy as np


class SpeculativeDecodingEngine:

    @staticmethod
    def simulate_verification(
        lookahead_k: int = 5,
        acceptance_rate: float = 0.80,
        num_trials: int = 100,
    ) -> Tuple[float, float]:
        """
        Simula el proceso de aceptación de prefijos especulativos.
        Calcula la longitud media del prefijo aceptado y la aceleración teórica.
        """
        accepted_lengths = []
        for _ in range(num_trials):
            # Probar cada token especulativo secuencialmente hasta el primer rechazo
            accepted = 0
            for _ in range(lookahead_k):
                if np.random.rand() <= acceptance_rate:
                    accepted += 1
                else:
                    break
            accepted_lengths.append(accepted)

        avg_accepted = float(np.mean(accepted_lengths))
        # Cada paso genera los tokens aceptados + 1 token corregido del modelo target
        effective_tokens_per_step = avg_accepted + 1.0
        speedup = effective_tokens_per_step / (1.0 + (lookahead_k * 0.1))  # Penalización por costo del draft

        logger.info(
            f"Speculative Decoding (K={lookahead_k}, Acceptance={acceptance_rate*100:.0f}%): "
            f"Mean Tokens/Step: {effective_tokens_per_step:.2f}x | Net Speedup: {speedup:.2f}x"
        )
        return round(avg_accepted, 2), round(speedup, 2)
