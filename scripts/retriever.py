#!/usr/bin/env python3
"""Retriever utilities for LIS dataset generation.

Questo modulo è il "pezzo R" di una pipeline RAG:
- prende una frase italiana,
- recupera le voci lessicali pertinenti dal vocabolario,
- recupera le sezioni di regole grammaticali più utili,
- costruisce un contesto/prompt compatto per la generazione.

Scelta progettuale: nessuna dipendenza esterna, solo standard library,
così il progetto resta eseguibile in ambienti minimali.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RetrievedContext:
    """Contenitore strutturato del contesto recuperato.

    Attributi:
        source_text: frase originale in italiano.
        sentence_type: categoria frase (declarative / polar_question).
        tokens: token normalizzati in minuscolo.
        detected_lemmas: lemmi LIS individuati tramite mapping lessicale.
        lexical_entries: sotto-dizionario del vocabolario utile per la frase.
        rule_sections: sezioni delle regole grammaticali selezionate.
    """

    source_text: str
    sentence_type: str
    tokens: list[str]
    detected_lemmas: list[str]
    lexical_entries: dict[str, dict[str, Any]]
    rule_sections: dict[str, str]


class LISRetriever:
    """Recupera lessico e regole rilevanti per una frase.

    Nota: il retriever attuale è volutamente semplice (heuristic-based),
    ma l'interfaccia è già pronta per essere estesa con retrieval semantico
    (BM25, embeddings, vector DB, ecc.).
    """

    # Regex permissiva per token alfabetici (inclusi caratteri accentati italiani).
    _TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ']+")

    # Mapping minimo token ITA -> lemma LIS (in maiuscolo, convenzione glosse).
    # Questo mapping è il ponte tra testo italiano e dizionario segni.
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
        """Carica una volta sola lessico e regole.

        Parameters:
            vocab_path: percorso a `data/vocabolario.json`.
            rules_path: percorso a `data/regole_grammatica.md`.
        """
        self.vocab_path = Path(vocab_path)
        self.rules_path = Path(rules_path)
        self.vocab = self._load_json(self.vocab_path)
        self.rules_text = self.rules_path.read_text(encoding="utf-8")
        # Pre-parsing delle sezioni markdown per retrieval più rapido.
        self.rules_sections = self._split_rules_sections(self.rules_text)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        """Utility centralizzata per leggere file JSON UTF-8."""
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _split_rules_sections(markdown_text: str) -> dict[str, str]:
        """Divide un markdown in sezioni H2 (`## titolo`).

        Restituisce:
            dict[titolo_sezione] = contenuto_testuale

        Perché serve:
            ci permette di recuperare solo il sottoinsieme di regole utili,
            invece di passare sempre tutto il documento al generatore.
        """
        sections: dict[str, str] = {}
        current_title = "INTRO"
        buff: list[str] = []

        for line in markdown_text.splitlines():
            if line.startswith("## "):
                # Salva la sezione precedente quando incontriamo un nuovo header.
                sections[current_title] = "\n".join(buff).strip()
                current_title = line[3:].strip()
                buff = []
            else:
                buff.append(line)

        # Salva anche l'ultima sezione accumulata.
        sections[current_title] = "\n".join(buff).strip()
        # Rimuove eventuali sezioni vuote.
        return {k: v for k, v in sections.items() if v}

    def tokenize(self, text: str) -> list[str]:
        """Tokenizza il testo in forma normalizzata (minuscolo)."""
        return [m.group(0).lower() for m in self._TOKEN_RE.finditer(text)]

    def detect_sentence_type(self, text: str) -> str:
        """Classifica tipo frase con euristica semplice sul punto interrogativo."""
        return "polar_question" if text.strip().endswith("?") else "declarative"

    def detect_lemmas(self, tokens: list[str]) -> list[str]:
        """Mappa i token ai lemmi LIS, rimuovendo duplicati e preservando ordine."""
        lemmas = []
        seen = set()
        for tok in tokens:
            lemma = self._TOKEN_TO_LEMMA.get(tok)
            if lemma and lemma not in seen:
                lemmas.append(lemma)
                seen.add(lemma)
        return lemmas

    def retrieve_lexicon(self, lemmas: list[str]) -> dict[str, dict[str, Any]]:
        """Recupera dal vocabolario solo le voci effettivamente necessarie."""
        return {lemma: self.vocab[lemma] for lemma in lemmas if lemma in self.vocab}

    def retrieve_rules(self, sentence_type: str, tokens: list[str]) -> dict[str, str]:
        """Seleziona le sezioni di regole pertinenti al fenomeno linguistico.

        Strategia:
        - include sempre coreferenza/riferimento spaziale;
        - include verbi flessivi se ci sono verbi target;
        - include sezione affermative o interrogative in base al tipo frase;
        - include sempre normalizzazione glosse e anti-allucinazione.
        """
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
        """Metodo principale del retriever.

        Flusso:
            testo -> token -> tipo frase -> lemmi -> lessico -> regole -> context.
        """
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
        """Costruisce un prompt compatto per un LLM fine-tuned.

        Il prompt include:
            1) task + vincolo JSON,
            2) frase + tipo,
            3) lessico recuperato,
            4) regole recuperate.

        Anche se la baseline attuale è deterministica, questo metodo è già
        utile per il passaggio futuro a inferenza reale con Minerva 7B.
        """
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
    # Demo locale del retriever:
    # - utile per verificare rapidamente tokenizzazione/retrieval/prompt.
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
