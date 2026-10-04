import re
from typing import List
from chapter02_data_collection.src.models import RawDocument


class TextCleaner:

    @staticmethod
    def clean(document: RawDocument) -> RawDocument:
        text = document.content

        # Eliminar URLs irrelevantes del cuerpo manteniendo formato
        text = re.sub(r"http\S+", "", text)
        # Normalizar saltos de línea y espacios repetidos
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text).strip()

        document.content = text
        return document

    @classmethod
    def clean_batch(cls, documents: List[RawDocument]) -> List[RawDocument]:
        cleaned = [cls.clean(doc) for doc in documents]
        return [doc for doc in cleaned if len(doc.content) > 15]
