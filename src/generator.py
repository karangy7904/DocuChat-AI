from __future__ import annotations

import re

from .pdf_processor import Chunk
from .prompts import build_messages


ABSTAIN = "I could not find this information in the supplied document."


class ExtractiveGenerator:
    """Fast deployment fallback that still demonstrates retrieval and citations."""

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return ABSTAIN

        normalized_question = question.lower()
        summary_request = any(
            phrase in normalized_question
            for phrase in ("summary", "summarize", "summarise", "overview", "main points")
        )

        if summary_request:
            excerpts = []
            seen = set()
            for chunk in chunks:
                sentences = [
                    sentence.strip()
                    for sentence in re.split(r"(?<=[.!?])\s+", chunk.text)
                    if len(sentence.strip().split()) >= 5
                ]
                if sentences:
                    sentence = sentences[0]
                    if sentence not in seen:
                        excerpts.append(f"{sentence} [Page {chunk.page}]")
                        seen.add(sentence)
                if len(excerpts) == 3:
                    break
            return " ".join(excerpts) if excerpts else ABSTAIN

        terms = set(re.findall(r"[a-z0-9]+", question.lower())) - {
            "a", "an", "are", "is", "of", "the", "to", "what", "which"
        }
        candidates = []
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                candidates.append((len(terms & words), len(sentence), sentence, chunk.page))
        if not candidates:
            return ABSTAIN
        _, _, sentence, page = max(candidates)
        # FAISS has already selected semantically similar evidence. Returning a
        # verbatim sentence remains grounded even when the query uses synonyms.
        return f"{sentence.strip()} [Page {page}]"


class TransformersGenerator:
    def __init__(self, model_id: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
        )
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            build_messages(question, chunks), tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        tokens = output[0][inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(tokens,skip_special_tokens=True).strip()
        if chunks and not re.search(r"\[Page \d+\]",answer):
            answer = f"{answer} [Page {chunks[0].page}]"
        return answer
