ARTICLE_INSTRUCTION_PROMPT = (
    "Synthesize the following article excerpt into a direct technical explanation or architectural guide."
)

POST_INSTRUCTION_PROMPT = (
    "Draft a concise, high-impact technical post reflecting on software systems and engineering best practices."
)

CODE_INSTRUCTION_PROMPT = (
    "Provide a production-ready Python implementation for the following system component."
)


def get_instruction_for_category(category: str) -> str:
    mapping = {
        "articles": ARTICLE_INSTRUCTION_PROMPT,
        "posts": POST_INSTRUCTION_PROMPT,
        "repositories": CODE_INSTRUCTION_PROMPT,
        "code": CODE_INSTRUCTION_PROMPT,
    }
    return mapping.get(category, "Write a technical analysis about this topic.")
