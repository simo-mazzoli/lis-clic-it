#!/usr/bin/env python3
"""Generate LIS annotated dataset records (JSONL).

Default mode is deterministic rule-based generation to keep the pipeline runnable
without external APIs. It still uses retrieval context so the design can be
extended to a real Minerva 7B inference call later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retriever import LISRetriever, RetrievedContext


@dataclass
class GenerationConfig:
    dataset_version: str
    model_name: str
    rules_version: str
    lexicon_version: str


class SimpleSchemaValidator:
    """Minimal validator aligned to our JSON schema required fields/types.

    This does not implement full JSON Schema draft-07. It checks the most
    relevant constraints needed by this project.
    """

    REQUIRED_TOP_LEVEL = {
        "id": str,
        "dataset_version": str,
        "source_text_it": str,
        "sentence_type": str,
        "lis_gloss": str,
        "signs": list,
        "references": list,
        "generation_meta": dict,
    }

    REQUIRED_SIGN = {
        "lemma": str,
        "form": str,
        "pos": str,
    }

    REQUIRED_REF = {
        "entity": str,
        "locus": str,
    }

    REQUIRED_META = {
        "model": str,
        "prompt_hash": str,
        "timestamp": str,
    }

    def validate(self, record: dict[str, Any]) -> tuple[bool, str]:
        for field, typ in self.REQUIRED_TOP_LEVEL.items():
            if field not in record:
                return False, f"Missing top-level field: {field}"
            if not isinstance(record[field], typ):
                return False, f"Invalid type for {field}: expected {typ.__name__}"

        if record["sentence_type"] not in {"declarative", "polar_question"}:
            return False, "sentence_type must be declarative or polar_question"

        if not record["signs"]:
            return False, "signs must contain at least one item"

        for i, sign in enumerate(record["signs"]):
            if not isinstance(sign, dict):
                return False, f"sign[{i}] is not an object"
            for field, typ in self.REQUIRED_SIGN.items():
                if field not in sign:
                    return False, f"Missing sign field {field} in sign[{i}]"
                if not isinstance(sign[field], typ):
                    return False, f"Invalid type for sign[{i}].{field}"

            confidence = sign.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    return False, f"Invalid confidence type in sign[{i}]"
                if not (0 <= float(confidence) <= 1):
                    return False, f"confidence out of range in sign[{i}]"

        for i, ref in enumerate(record["references"]):
            if not isinstance(ref, dict):
                return False, f"references[{i}] is not an object"
            for field, typ in self.REQUIRED_REF.items():
                if field not in ref or not isinstance(ref[field], typ):
                    return False, f"Invalid or missing references[{i}].{field}"

        meta = record["generation_meta"]
        for field, typ in self.REQUIRED_META.items():
            if field not in meta or not isinstance(meta[field], typ):
                return False, f"Invalid or missing generation_meta.{field}"

        return True, "ok"


class LISGenerator:
    def __init__(self, retriever: LISRetriever, schema_path: Path, cfg: GenerationConfig):
        self.retriever = retriever
        self.schema_path = Path(schema_path)
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.validator = SimpleSchemaValidator()

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _detect_entities(tokens: list[str]) -> list[str]:
        entities = []
        for t in tokens:
            if t == "maria" and "MARIA" not in entities:
                entities.append("MARIA")
            if t == "luca" and "LUCA" not in entities:
                entities.append("LUCA")
        return entities

    @staticmethod
    def _build_references(entities: list[str]) -> list[dict[str, str]]:
        loci = ["A", "B", "C", "D"]
        return [{"entity": ent, "locus": loci[i]} for i, ent in enumerate(entities)]

    @staticmethod
    def _normalize_source(text: str) -> str:
        return " ".join(text.strip().split())

    def _token_to_lemma(self, token: str) -> str | None:
        mapping = LISRetriever._TOKEN_TO_LEMMA
        return mapping.get(token)

    def _make_sign(
        self,
        lemma: str,
        lexical_entries: dict[str, dict[str, Any]],
        sentence_type: str,
        agreement: dict[str, str] | None = None,
        topic: bool | None = None,
        oov: bool = False,
    ) -> dict[str, Any]:
        lex = lexical_entries.get(lemma, {})
        sign: dict[str, Any] = {
            "lemma": lemma,
            "form": lemma,
            "pos": lex.get("tipo_grammaticale", "sconosciuto"),
            "location": lex.get("luogo", "unknown"),
            "handshape": lex.get("configurazione", "unknown"),
            "movement": lex.get("movimento", "unknown"),
            "orientation": lex.get("orientamento", "unknown"),
            "oov": oov,
            "confidence": 0.99 if not oov else 0.4,
        }

        if sentence_type == "polar_question":
            sign["non_manuals"] = ["eyebrows_raised"]

        if agreement:
            sign["agreement"] = agreement

        if topic is not None:
            sign["topic"] = topic

        return sign

    def _extract_agreement(
        self, tokens: list[str], references: list[dict[str, str]], lemma: str
    ) -> dict[str, str] | None:
        ref_map = {r["entity"]: r["locus"] for r in references}

        # Heuristic for sample domain.
        if lemma in {"VEDERE", "DARE"}:
            if "maria" in tokens and "luca" in tokens:
                if tokens.index("maria") < tokens.index("luca"):
                    return {
                        "source_locus": ref_map.get("MARIA", "A"),
                        "target_locus": ref_map.get("LUCA", "B"),
                    }
                return {
                    "source_locus": ref_map.get("LUCA", "A"),
                    "target_locus": ref_map.get("MARIA", "B"),
                }
        return None

    def generate_record(self, source_text: str, idx: int) -> dict[str, Any]:
        source_text = self._normalize_source(source_text)
        context: RetrievedContext = self.retriever.retrieve(source_text)
        prompt = self.retriever.build_prompt(context)
        prompt_hash = self._sha256(prompt)

        entities = self._detect_entities(context.tokens)
        references = self._build_references(entities)

        signs: list[dict[str, Any]] = []
        for token in context.tokens:
            lemma = self._token_to_lemma(token)
            if not lemma:
                continue
            agreement = self._extract_agreement(context.tokens, references, lemma)
            signs.append(
                self._make_sign(
                    lemma=lemma,
                    lexical_entries=context.lexical_entries,
                    sentence_type=context.sentence_type,
                    agreement=agreement,
                    topic=True if lemma in {"DOMANI"} else None,
                    oov=lemma not in context.lexical_entries,
                )
            )

        # Deduplicate consecutive repeated lemmas if any.
        deduped_signs: list[dict[str, Any]] = []
        for sign in signs:
            if not deduped_signs or deduped_signs[-1]["lemma"] != sign["lemma"]:
                deduped_signs.append(sign)

        lis_gloss = " ".join(s["form"] for s in deduped_signs)

        record = {
            "id": f"lis-{idx:06d}",
            "dataset_version": self.cfg.dataset_version,
            "source_text_it": source_text,
            "sentence_type": context.sentence_type,
            "lis_gloss": lis_gloss,
            "signs": deduped_signs,
            "references": references,
            "notes": "auto-generated (deterministic baseline)",
            "generation_meta": {
                "model": self.cfg.model_name,
                "prompt_hash": prompt_hash,
                "timestamp": self._now_iso(),
                "retrieved_rules_version": self.cfg.rules_version,
                "retrieved_lexicon_version": self.cfg.lexicon_version,
            },
        }

        is_valid, msg = self.validator.validate(record)
        if not is_valid:
            raise ValueError(f"Record validation failed for id={record['id']}: {msg}")

        return record

    def generate_dataset(self, input_lines: list[str]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for idx, line in enumerate(input_lines, start=1):
            if not line.strip():
                continue
            records.append(self.generate_record(line.strip(), idx=idx))
        return records


def read_input_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LIS JSONL dataset.")
    parser.add_argument("--input", default="data/input_frasi.txt", help="Path to input sentences")
    parser.add_argument("--vocab", default="data/vocabolario.json", help="Path to vocabulary JSON")
    parser.add_argument("--rules", default="data/regole_grammatica.md", help="Path to grammar rules")
    parser.add_argument("--schema", default="data/schema_annotazione.json", help="Path to JSON schema")
    parser.add_argument("--output", default="output/dataset_lis_final.jsonl", help="Output JSONL path")
    parser.add_argument("--dataset-version", default="v0.1.0")
    parser.add_argument("--model-name", default="minerva-7b-finetuned")
    parser.add_argument("--rules-version", default="regole_grammatica@1")
    parser.add_argument("--lexicon-version", default="vocabolario@1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    input_path = root / args.input
    vocab_path = root / args.vocab
    rules_path = root / args.rules
    schema_path = root / args.schema
    output_path = root / args.output

    retriever = LISRetriever(vocab_path=vocab_path, rules_path=rules_path)
    cfg = GenerationConfig(
        dataset_version=args.dataset_version,
        model_name=args.model_name,
        rules_version=args.rules_version,
        lexicon_version=args.lexicon_version,
    )

    generator = LISGenerator(retriever=retriever, schema_path=schema_path, cfg=cfg)
    lines = read_input_lines(input_path)
    records = generator.generate_dataset(lines)
    write_jsonl(output_path, records)

    print(f"Generated {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
