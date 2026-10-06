from chapter09_rag_inference_pipeline.src.rag.base import PromptTemplateFactory


class QueryExpansionTemplate(PromptTemplateFactory):
    separator: str = "#next-question#"

    def create_template(self, expand_to_n: int) -> str:
        return (
            f"You are an expert AI assistant. Generate {expand_to_n} different versions of the following "
            f"user question to retrieve relevant context across multiple viewpoints from a vector database.\n"
            f"Separate questions with '{self.separator}'.\n"
            f"Original Question: {{question}}"
        )


class SelfQueryTemplate(PromptTemplateFactory):
    def create_template(self) -> str:
        return (
            "Extract the author or user name from the following question. "
            "Return ONLY the extracted name, or 'none' if no author name is present.\n"
            "Question: {question}"
        )


class RAGPromptTemplate(PromptTemplateFactory):
    def create_template(self) -> str:
        return (
            "You are Oscar's LLM Twin, an expert AI and Data Architect. "
            "Write what the user asked you while using the provided context as the primary source of truth. "
            "Maintain an authentic, direct, and technically thorough voice without generic pleasantries.\n\n"
            "Context:\n{context}\n\n"
            "User query: {query}\n\n"
            "Answer:"
        )
