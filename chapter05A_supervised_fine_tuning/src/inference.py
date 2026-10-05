from loguru import logger
from chapter05A_supervised_fine_tuning.src.chat_templates import ALPACA_TEMPLATE


class TwinInferenceRunner:

    @staticmethod
    def generate(instruction: str) -> str:
        logger.info(f"Running inference with instruction: '{instruction}'")
        
        prompt = ALPACA_TEMPLATE.format(instruction=instruction, output="")
        
        # Simulación de respuesta generada por el Twin fine-tuneado
        response = (
            "Supervised Fine-Tuning adapts an autoregressive LLM to reproduce the user's specific "
            "reasoning style and format conventions, while RAG grounds factual correctness dynamically."
        )

        print("\n" + "=" * 50)
        print("INFERENCE GENERATION (Alpaca Formatted):")
        print(prompt + response)
        print("=" * 50 + "\n")

        return response
