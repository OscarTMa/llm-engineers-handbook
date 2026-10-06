import json
from pathlib import Path
from loguru import logger
import yaml

from chapter06_preference_alignment.src.dataset_generator import PreferenceDatasetBuilder
from chapter06_preference_alignment.src.dpo_trainer import DPOTrainerEngine
from chapter06_preference_alignment.src.inference import PreferenceInferenceComparator


def main():
    logger.info("=== Starting Chapter 06: Preference Alignment (DPO) ===")

    # 1. Cargar configuraciones
    cfg_path = "chapter06_preference_alignment/configs/dpo.yaml"
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Sintetizar triples (prompt, chosen, rejected)
    raw_path = config["storage"]["raw_data_dir"]
    raw_triples = PreferenceDatasetBuilder.synthesize_triples_from_raw(raw_path)

    # 3. Aplicar filtros heurísticos de calidad y formato
    curated_triples = PreferenceDatasetBuilder.filter_short_answers(
        raw_triples, min_length=config["curation"]["min_chosen_length"]
    )
    if config["curation"]["enforce_punctuation"]:
        curated_triples = PreferenceDatasetBuilder.filter_answer_format(curated_triples)

    logger.success(f"Curated {len(curated_triples)} high-quality preference triples.")

    # 4. Guardar dataset procesado en disco
    out_file = Path(config["storage"]["output_dataset_path"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for item in curated_triples:
            f.write(json.dumps(item.model_dump(), ensure_ascii=False) + "\n")
    logger.info(f"Persisted DPO preference dataset to: {out_file}")

    # 5. Ejecutar alineación DPO
    trainer = DPOTrainerEngine(config)
    trainer.train(curated_triples)

    # 6. Registrar adaptador alineado
    adapter_dir = Path(config["storage"]["output_adapter_dir"])
    adapter_dir.mkdir(parents=True, exist_ok=True)
    with open(adapter_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(config["lora"], f, indent=2)
    logger.success(f"DPO aligned adapter registered at: {adapter_dir}")

    # 7. Comparación de salida (SFT vs DPO)
    PreferenceInferenceComparator.compare("Introduce supervised fine-tuning in the context of an LLM Twin.")

    logger.success("=== Chapter 06 Completed Successfully ===")


if __name__ == "__main__":
    main()
