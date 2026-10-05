import json
from pathlib import Path
from loguru import logger
import yaml

from chapter05A_supervised_fine_tuning.src.chat_templates import ChatTemplateFormatter
from chapter05A_supervised_fine_tuning.src.dataset_generator import InstructionDatasetBuilder
from chapter05A_supervised_fine_tuning.src.inference import TwinInferenceRunner
from chapter05A_supervised_fine_tuning.src.trainer import SFTTrainerEngine


def main():
    logger.info("=== Starting Chapter 05A: Supervised Fine-Tuning Pipeline ===")

    # 1. Cargar configuraciones
    cfg_path = "chapter05A_supervised_fine_tuning/configs/sft.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Sintetizar dataset de instrucciones desde los artículos brutos
    raw_path = config["storage"]["raw_articles_path"]
    dataset_records = InstructionDatasetBuilder.build_from_raw_warehouse(raw_path)

    # 3. Formatear muestras con la plantilla de chat Alpaca y delimitador EOS
    formatted_dataset = ChatTemplateFormatter.apply_formatting(dataset_records)

    # 4. Exportar el dataset procesado
    out_file = Path(config["storage"]["output_dataset_path"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for item in dataset_records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Persisted instruction dataset to: {out_file}")

    # 5. Ejecutar entrenamiento SFT con LoRA
    trainer = SFTTrainerEngine(config)
    trainer.train(formatted_dataset)

    # 6. Guardar metadatos del adaptador
    adapter_dir = Path(config["storage"]["output_adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config["lora"], f, indent=2)
    logger.success(f"Adapter checkpoints and configuration registered at: {adapter_dir}")

    # 7. Probar inferencia en el modelo fine-tuneado
    TwinInferenceRunner.generate("Explain how Supervised Fine-Tuning works for an LLM Twin.")

    logger.success("=== Chapter 05A Completed Successfully ===")


if __name__ == "__main__":
    main()
