import re
from typing import Tuple
from loguru import logger


class SafetyGuardrails:

    # Patrones regex para detección de filtración de PII y credenciales
    PII_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL_MASKED]"),
        (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE_MASKED]"),
        (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "[AWS_ACCESS_KEY_MASKED]"),
    ]

    JAILBREAK_TRIGGERS = [
        "ignore previous instructions",
        "system prompt override",
        "drop table",
        "delete database",
        "act as a malicious",
    ]

    @classmethod
    def sanitize_input(cls, query: str) -> Tuple[bool, str]:
        """
        Valida y desinfecta el prompt de entrada del usuario.
        Retorna (is_allowed, sanitized_query).
        """
        lower_q = query.lower()

        # 1. Detección de Prompt Injection / Jailbreaking
        for trigger in cls.JAILBREAK_TRIGGERS:
            if trigger in lower_q:
                logger.warning(f"Guardrail Alert: Malicious prompt injection trigger detected -> '{trigger}'")
                return False, "Input rejected: Prompt violates system security guidelines."

        # 2. Desinfección de Información Privada (PII)
        sanitized = query
        for pattern, replacement in cls.PII_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)

        if sanitized != query:
            logger.info("Guardrail Notice: Sensitive PII patterns were masked from input.")

        return True, sanitized

    @classmethod
    def sanitize_output(cls, output: str) -> Tuple[bool, str]:
        """
        Valida que el texto generado no contenga credenciales filtradas ni respuestas vacías.
        """
        if not output or len(output.strip()) == 0:
            return False, "Output rejected: Model generated empty response."

        sanitized = output
        for pattern, replacement in cls.PII_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)

        return True, sanitized
