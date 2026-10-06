from typing import Dict, List
from loguru import logger
from pydantic import BaseModel


class InferenceRequest(BaseModel):
    request_id: str
    prompt_tokens: int
    generated_tokens_target: int
    current_tokens: int = 0
    completed: bool = False


class ContinuousBatchingScheduler:

    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.active_batch: List[InferenceRequest] = []
        self.waiting_queue: List[InferenceRequest] = []
        self.completed_requests: List[InferenceRequest] = []

    def add_requests(self, requests: List[InferenceRequest]):
        self.waiting_queue.extend(requests)

    def step(self) -> int:
        """
        Ejecuta un paso de decodificación iterativo.
        Evita burbujas desalojando solicitudes completadas e insertando nuevas inmediatamente.
        """
        # 1. Rellenar ranuras libres desde la cola de espera
        while len(self.active_batch) < self.max_batch_size and self.waiting_queue:
            new_req = self.waiting_queue.pop(0)
            self.active_batch.append(new_req)

        # 2. Generar 1 token por solicitud activa
        newly_completed = []
        for req in self.active_batch:
            req.current_tokens += 1
            if req.current_tokens >= req.generated_tokens_target:
                req.completed = True
                newly_completed.append(req)

        # 3. Desalojar solicitudes finalizadas
        for comp in newly_completed:
            self.active_batch.remove(comp)
            self.completed_requests.append(comp)

        return len(self.active_batch)

    def run_simulation(self) -> Dict[str, float]:
        step_count = 0
        while self.active_batch or self.waiting_queue:
            step_count += 1
            self.step()

        total_tokens_generated = sum(r.generated_tokens_target for r in self.completed_requests)
        throughput_rate = round(total_tokens_generated / max(step_count, 1), 2)

        logger.success(
            f"Continuous Batching Simulation: Processed {len(self.completed_requests)} requests "
            f"in {step_count} steps (Throughput: {throughput_rate} tokens/step)"
        )
        return {
            "total_requests": len(self.completed_requests),
            "total_steps": step_count,
            "throughput_tokens_per_step": throughput_rate,
        }
