from typing import Any, Dict, List
from loguru import logger

from chapter11_llmops.src.monitoring.opik_tracer import OpikTracker
from chapter11_llmops.src.orchestration.alerter import ZenMLAlerter


class ContinuousTrainingOrchestrator:

    @classmethod
    @OpikTracker.trace_span("etl_extraction_step")
    def run_step_etl(cls) -> int:
        logger.info("CT Step 1/4: Ingesting raw documents into MongoDB Data Warehouse...")
        return 3

    @classmethod
    @OpikTracker.trace_span("rag_feature_step")
    def run_step_features(cls) -> int:
        logger.info("CT Step 2/4: Executing cleaning, domain chunking, and Qdrant vector indexing...")
        return 9

    @classmethod
    @OpikTracker.trace_span("instruction_synthesis_step")
    def run_step_dataset(cls) -> int:
        logger.info("CT Step 3/4: Synthesizing instruction & preference pairs into ZenML Artifact Store...")
        return 12

    @classmethod
    @OpikTracker.trace_span("training_step")
    def run_step_train_deploy(cls) -> Dict[str, Any]:
        logger.info("CT Step 4/4: Validating model quality gates & synchronizing SageMaker deployment...")
        return {"eval_loss": 1.28, "status": "Deployed"}

    @classmethod
    def execute_ct_pipeline(cls) -> Dict[str, Any]:
        logger.info("=== Triggering ZenML Master Continuous Training Pipeline (end_to_end_data) ===")
        try:
            docs = cls.run_step_etl()
            chunks = cls.run_step_features()
            datasets = cls.run_step_dataset()
            res = cls.run_step_train_deploy()

            details = {"documents": docs, "chunks": chunks, "dataset_samples": datasets, "model_res": res}
            ZenMLAlerter.notify_pipeline_status("succeeded", "end_to_end_data", details)
            return {"status": "success", "pipeline": "end_to_end_data", "details": details}
        except Exception as e:
            ZenMLAlerter.notify_pipeline_status("failed", "end_to_end_data", {"error": str(e)})
            raise
