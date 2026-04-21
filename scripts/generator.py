#!/usr/bin/env python3
"""Generate LIS annotated dataset records (JSONL).

Questo script implementa una baseline deterministica, quindi:
- il progetto è subito eseguibile senza endpoint esterni,
- la struttura è compatibile con una futura integrazione LLM reale.

Pipeline:
1) legge frasi italiane,
2) usa il retriever per contesto lessico/regole,
3) produce record LIS annotati,
4) valida i record,
5) salva JSONL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from retriever import LISRetriever, RetrievedContext


@dataclass
class GenerationConfig:
    """Configurazione versionata del processo di generazione.

    Questi campi finiscono nei metadati dei record per tracciabilità.
    """

    dataset_version: str
    model_name: str
    hf_token: str | None
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

    # Campi minimi richiesti per ogni segno.
    REQUIRED_SIGN = {
        "lemma": str,
        "form": str,
        "pos": str,
    }

    # Campi minimi per referenze/loci.
    REQUIRED_REF = {
        "entity": str,
        "locus": str,
    }

    # Campi minimi metadati generazione.
    REQUIRED_META = {
        "model": str,
        "prompt_hash": str,
        "timestamp": str,
    }

    def validate(self, record: dict[str, Any]) -> tuple[bool, str]:
        """Valida il record e restituisce (ok, messaggio)."""
        # 1) Check top-level.
        for field, typ in self.REQUIRED_TOP_LEVEL.items():
            if field not in record:
                return False, f"Missing top-level field: {field}"
            if not isinstance(record[field], typ):
                return False, f"Invalid type for {field}: expected {typ.__name__}"

        # 2) Check dominio sentence_type.
        if record["sentence_type"] not in {"declarative", "polar_question"}:
            return False, "sentence_type must be declarative or polar_question"

        # 3) signs deve esistere e non essere vuoto.
        if not record["signs"]:
            return False, "signs must contain at least one item"

        # 4) Check ogni segno.
        for i, sign in enumerate(record["signs"]):
            if not isinstance(sign, dict):
                return False, f"sign[{i}] is not an object"
            for field, typ in self.REQUIRED_SIGN.items():
                if field not in sign:
                    return False, f"Missing sign field {field} in sign[{i}]"
                if not isinstance(sign[field], typ):
                    return False, f"Invalid type for sign[{i}].{field}"

            # Confidence opzionale ma, se presente, deve stare in [0,1].
            confidence = sign.get("confidence")
            if confidence is not None:
                if not isinstance(confidence, (int, float)):
                    return False, f"Invalid confidence type in sign[{i}]"
                if not (0 <= float(confidence) <= 1):
                    return False, f"confidence out of range in sign[{i}]"

        # 5) Check references.
        for i, ref in enumerate(record["references"]):
            if not isinstance(ref, dict):
                return False, f"references[{i}] is not an object"
            for field, typ in self.REQUIRED_REF.items():
                if field not in ref or not isinstance(ref[field], typ):
                    return False, f"Invalid or missing references[{i}].{field}"

        # 6) Check generation meta.
        meta = record["generation_meta"]
        for field, typ in self.REQUIRED_META.items():
            if field not in meta or not isinstance(meta[field], typ):
                return False, f"Invalid or missing generation_meta.{field}"

        return True, "ok"


class LISGenerator:
    """Generatore di record LIS.

    Usa retriever + euristiche deterministiche per ottenere un baseline dataset
    coerente e riproducibile.
    """

    def __init__(self, retriever: LISRetriever, schema_path: Path, cfg: GenerationConfig):
        self.retriever = retriever
        self.schema_path = Path(schema_path)
        # Carichiamo lo schema per tracciabilità/documentazione interna.
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.cfg = cfg
        self.validator = SimpleSchemaValidator()

    @staticmethod
    def _sha256(text: str) -> str:
        """Hash stabile del prompt (auditability)."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_iso() -> str:
        """Timestamp ISO8601 UTC per metadati."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _detect_entities(tokens: list[str]) -> list[str]:
        """Estrae entità nominali note mantenendo ordine di apparizione."""
        entities = []
        for t in tokens:
            if t == "maria" and "MARIA" not in entities:
                entities.append("MARIA")
            if t == "luca" and "LUCA" not in entities:
                entities.append("LUCA")
        return entities

    @staticmethod
    def _build_references(entities: list[str]) -> list[dict[str, str]]:
        """Assegna loci A/B/C/D ai referenti in ordine.

        Esempio: [MARIA, LUCA] -> [{MARIA:A}, {LUCA:B}]
        """
        loci = ["A", "B", "C", "D"]
        return [{"entity": ent, "locus": loci[i]} for i, ent in enumerate(entities)]

    @staticmethod
    def _normalize_source(text: str) -> str:
        """Normalizza spazi extra ma preserva il contenuto lessicale."""
        return " ".join(text.strip().split())

    def _token_to_lemma(self, token: str) -> str | None:
        """Riusa il mapping centralizzato del retriever per coerenza globale."""
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
        """Costruisce l'oggetto `sign` completo.

        Se il lemma è OOV, inserisce placeholder + confidence più bassa.
        """
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

        # Per interrogative polari aggiungiamo marcatore non-manuale standard.
        if sentence_type == "polar_question":
            sign["non_manuals"] = ["eyebrows_raised"]

        # Verbi direzionali/flessivi: accordo source->target.
        if agreement:
            sign["agreement"] = agreement

        # Topic opzionale (es. DOMANI topicalizzato).
        if topic is not None:
            sign["topic"] = topic

        return sign

    def _extract_agreement(
        self, tokens: list[str], references: list[dict[str, str]], lemma: str
    ) -> dict[str, str] | None:
        """Inferisce accordo base per verbi target (euristica).

        Regola usata nel dominio seed:
        - se frase contiene MARIA e LUCA,
        - direzione dipende dall'ordine di comparsa nella frase italiana.
        """
        ref_map = {r["entity"]: r["locus"] for r in references}

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
        """Genera un record completo e validato a partire da una frase."""
        source_text = self._normalize_source(source_text)

        # Recuperiamo il contesto RAG (lessico + regole) per questa frase.
        context: RetrievedContext = self.retriever.retrieve(source_text)

        # Hashiamo il prompt per tracciare la "ricetta" di generazione.
        prompt = self.retriever.build_prompt(context)
        prompt_hash = self._sha256(prompt)

        # Costruiamo tabella referenti/loci.
        entities = self._detect_entities(context.tokens)
        references = self._build_references(entities)

        # Trasformiamo i token in segni annotati.
        signs: list[dict[str, Any]] = []
        for token in context.tokens:
            lemma = self._token_to_lemma(token)
            if not lemma:
                # Token ignorati: articoli/preposizioni/particelle non mappate.
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

        # Pulizia: evita duplicati consecutivi accidentali.
        deduped_signs: list[dict[str, Any]] = []
        for sign in signs:
            if not deduped_signs or deduped_signs[-1]["lemma"] != sign["lemma"]:
                deduped_signs.append(sign)

        # Glossa finale come sequenza forme.
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
                "hf_token_configured": bool(self.cfg.hf_token),
                "prompt_hash": prompt_hash,
                "timestamp": self._now_iso(),
                "retrieved_rules_version": self.cfg.rules_version,
                "retrieved_lexicon_version": self.cfg.lexicon_version,
            },
        }

        # Quality gate finale: record non valido => errore esplicito.
        is_valid, msg = self.validator.validate(record)
        if not is_valid:
            raise ValueError(f"Record validation failed for id={record['id']}: {msg}")

        return record

    def generate_dataset(self, input_lines: list[str]) -> list[dict[str, Any]]:
        """Genera una lista di record (uno per riga non vuota)."""
        records: list[dict[str, Any]] = []
        for idx, line in enumerate(input_lines, start=1):
            if not line.strip():
                continue
            records.append(self.generate_record(line.strip(), idx=idx))
        return records


def read_input_lines(path: Path) -> list[str]:
    """Legge file input e restituisce lista righe."""
    return path.read_text(encoding="utf-8").splitlines()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Scrive record in formato JSONL (1 JSON per riga)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    """Definisce interfaccia CLI dello script."""
    parser = argparse.ArgumentParser(description="Generate LIS JSONL dataset.")
    parser.add_argument("--input", default="data/input_frasi.txt", help="Path to input sentences")
    parser.add_argument("--vocab", default="data/vocabolario.json", help="Path to vocabulary JSON")
    parser.add_argument("--rules", default="data/regole_grammatica.md", help="Path to grammar rules")
    parser.add_argument("--schema", default="data/schema_annotazione.json", help="Path to JSON schema")
    parser.add_argument("--output", default="output/dataset_lis_final.jsonl", help="Output JSONL path")
    parser.add_argument("--dataset-version", default="v0.1.0")
    parser.add_argument("--model-name", default="sapienzanlp/Minerva-7B-instruct-v1.0")
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HUGGINGFACE_TOKEN"),
        help="Hugging Face token (default: env HUGGINGFACE_TOKEN)",
    )
    parser.add_argument("--rules-version", default="regole_grammatica@1")
    parser.add_argument("--lexicon-version", default="vocabolario@1")
    return parser.parse_args()


def main() -> None:
    """Entry-point CLI."""
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    # Risoluzione percorsi relativi alla root repo.
    input_path = root / args.input
    vocab_path = root / args.vocab
    rules_path = root / args.rules
    schema_path = root / args.schema
    output_path = root / args.output

    # Inizializza componenti pipeline.
    retriever = LISRetriever(vocab_path=vocab_path, rules_path=rules_path)
    cfg = GenerationConfig(
        dataset_version=args.dataset_version,
        model_name=args.model_name,
        hf_token=args.hf_token,
        rules_version=args.rules_version,
        lexicon_version=args.lexicon_version,
    )

    generator = LISGenerator(retriever=retriever, schema_path=schema_path, cfg=cfg)

    # Esecuzione generazione.
    lines = read_input_lines(input_path)
    records = generator.generate_dataset(lines)
    write_jsonl(output_path, records)

    print(f"Generated {len(records)} records -> {output_path}")


if __name__ == "__main__":
    main()
