from typing import Dict, List

ALPACA_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


class ChatTemplateFormatter:

    @staticmethod
    def format_alpaca(instruction: str, output: str, eos_token: str = "<|end_of_text|>") -> str:
        formatted = ALPACA_TEMPLATE.format(instruction=instruction, output=output)
        return f"{formatted}{eos_token}"

    @classmethod
    def apply_formatting(cls, samples: List[Dict[str, str]], eos_token: str = "<|end_of_text|>") -> List[Dict[str, str]]:
        formatted_dataset = []
        for s in samples:
            text = cls.format_alpaca(s["instruction"], s["output"], eos_token=eos_token)
            formatted_dataset.append({"text": text})
        return formatted_dataset
