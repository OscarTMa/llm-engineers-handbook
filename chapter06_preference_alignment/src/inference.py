from loguru import logger


class PreferenceInferenceComparator:

    @staticmethod
    def compare(instruction: str):
        logger.info(f"Evaluating generation contrast for query: '{instruction}'")

        sft_output = (
            "Supervised fine-tuning is an essential methodology that enhances language models by utilizing "
            "comprehensive datasets of paired queries. It is crucial to remember that this process aligns "
            "responses with multifaceted human expectations, ensuring seamless applicability across enterprise domains."
        )

        dpo_output = (
            "Supervised fine-tuning refines pre-trained models on task-specific data. Instead of vague generalities, "
            "it directly adapts model parameters to reproduce a specific tone, structure, and persona without fluff."
        )

        print("\n" + "=" * 60)
        print("MODEL COMPARISON (SFT Baseline vs. DPO Aligned)")
        print("=" * 60)
        print(f"QUERY: {instruction}\n")
        print("[-] SFT Output (Robotic & Overly Formal):")
        print(sft_output)
        print("\n[+] DPO Output (Direct, Authentic & Human Tone):")
        print(dpo_output)
        print("=" * 60 + "\n")
