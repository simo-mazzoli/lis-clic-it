#!/usr/bin/env python3
"""Retriever utilities for LIS dataset generation.

This module provides:
- lightweight lexical retrieval from `data/vocabolario.json`
- rule section retrieval from `data/regole_grammatica.md`
- prompt/context assembly for downstream generation

No external dependencies are required.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievedContext:
    """Container with all context needed by the generator."""

    source_text: str
    sentence_type: str
    tokens: list[str]
    detected_lemmas: list[str]
    lexical_entries: dict[str, dict[str, Any]]
    rule_sections: dict[str, str]


class LISRetriever:
    """Retrieve lexicon and relevant grammar rules for a sentence."""

    _TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

    # Very small normalization map for input seeds.
    _TOKEN_TO_LEMMA = {
        "maria": "MARIA",
        "luca": "LUCA",
        "mela": "MELA",
        "vede": "VEDERE",
        "vedere": "VEDERE",
        "dà": "DARE",
        "da": "DARE",
        "dare": "DARE",
        "saluta": "SALUTARE",
        "salutare": "SALUTARE",
        "domani": "DOMANI",
    }

    def __init__(self, vocab_path: Path, rules_path: Path) -> None:
        self.vocab_path = Path(vocab_path)
        self.rules_path = Path(rules_path)
        self.vocab = self._load_json(self.vocab_path)
        self.rules_text = self.rules_path.read_text(encoding="utf-8")
        self.rules_sections = self._split_rules_sections(self.rules_text)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _split_rules_sections(markdown_text: str) -> dict[str, str]:
        """Split markdown by h2 sections into a dictionary."""
        sections: dict[str, str] = {}
        current_title = "INTRO"
        buff: list[str] = []

        for line in markdown_text.splitlines():
            if line.startswith("## "):
                sections[current_title] = "\n".join(buff).strip()
                current_title = line[3:].strip()
                buff = []
            else:
                buff.append(line)

        sections[current_title] = "\n".join(buff).strip()
        return {k: v for k, v in sections.items() if v}

    def tokenize(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in self._TOKEN_RE.finditer(text)]

    def detect_sentence_type(self, text: str) -> str:
        return "polar_question" if text.strip().endswith("?") else "declarative"

    def detect_lemmas(self, tokens: list[str]) -> list[str]:
        lemmas = []
        seen = set()
        for tok in tokens:
            lemma = self._TOKEN_TO_LEMMA.get(tok)
            if lemma and lemma not in seen:
                lemmas.append(lemma)
                seen.add(lemma)
        return lemmas

    def retrieve_lexicon(self, lemmas: list[str]) -> dict[str, dict[str, Any]]:
        return {lemma: self.vocab[lemma] for lemma in lemmas if lemma in self.vocab}

    def retrieve_rules(self, sentence_type: str, tokens: list[str]) -> dict[str, str]:
        """Select only relevant sections based on detected phenomena."""
        selected: dict[str, str] = {}

        for title, content in self.rules_sections.items():
            ltitle = title.lower()
            if "riferimento spaziale" in ltitle:
                selected[title] = content
            elif "verbi flessivi" in ltitle and any(
                t in {"vede", "dà", "da", "saluta"} for t in tokens
            ):
                selected[title] = content
            elif "frasi affermative" in ltitle and sentence_type == "declarative":
                selected[title] = content
            elif "interrogative polari" in ltitle and sentence_type == "polar_question":
                selected[title] = content
            elif "normalizzazione glosse" in ltitle:
                selected[title] = content
            elif "anti-allucinazione" in ltitle:
                selected[title] = content

        return selected

    def retrieve(self, text: str) -> RetrievedContext:
        tokens = self.tokenize(text)
        sentence_type = self.detect_sentence_type(text)
        lemmas = self.detect_lemmas(tokens)
        lexical_entries = self.retrieve_lexicon(lemmas)
        rule_sections = self.retrieve_rules(sentence_type, tokens)

        return RetrievedContext(
            source_text=text,
            sentence_type=sentence_type,
            tokens=tokens,
            detected_lemmas=lemmas,
            lexical_entries=lexical_entries,
            rule_sections=rule_sections,
        )

    @staticmethod
    def build_prompt(context: RetrievedContext) -> str:
        """Build a compact prompt that could be sent to a fine-tuned LLM."""
        rules_block = "\n\n".join(
            f"[{title}]\n{content}" for title, content in context.rule_sections.items()
        )
        lex_block = json.dumps(context.lexical_entries, ensure_ascii=False, indent=2)

        return (
            "Sei un annotatore LIS. Rispondi solo con un JSON valido.\n\n"
            f"Frase IT: {context.source_text}\n"
            f"Tipo frase: {context.sentence_type}\n\n"
            f"LESSICO:\n{lex_block}\n\n"
            f"REGOLE:\n{rules_block}\n"
        )


if __name__ == "__main__":
    base = Path(__file__).resolve().parents[1]
    retriever = LISRetriever(
        vocab_path=base / "data" / "vocabolario.json",
        rules_path=base / "data" / "regole_grammatica.md",
    )

    sample = "Maria vede Luca e poi lo saluta."
    ctx = retriever.retrieve(sample)
    print(json.dumps(ctx.__dict__, ensure_ascii=False, indent=2))
    print("\n--- PROMPT ---\n")
    print(retriever.build_prompt(ctx))
