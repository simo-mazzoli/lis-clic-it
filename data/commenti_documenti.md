# Commenti esplicativi dei documenti

Questo file spiega **come leggere** ogni documento del progetto e il ruolo di ciascuna sezione.

---

## `README.md`

- **Obiettivo**: definisce il perché del progetto (dataset LIS per fine-tuning).
- **Perimetro linguistico**: chiarisce che si lavora su un subset, non su tutta la LIS.
- **Struttura repository**: mappa cartelle/file per orientarsi rapidamente.
- **Pipeline end-to-end**: ordine operativo consigliato (curazione, RAG, validazione, export).
- **Principi qualità**: regole per robustezza sperimentale (tracciabilità, bilanciamento, split).
- **Esecuzione rapida**: comandi minimi per eseguire subito il progetto.

## `data/regole_grammatica.md`

- Documento normativo operativo per la generazione.
- Ogni sezione corrisponde a un fenomeno target:
  - coreferenza/loci,
  - verbi flessivi,
  - affermative,
  - interrogative polari,
  - topicalizzazione,
  - normalizzazione glosse,
  - anti-allucinazione.
- Va usato come base per retrieval di regole in prompt RAG.

## `data/linee_guida_rag.md`

- Spiega *come* costruire il contesto del prompt:
  - lessico rilevante,
  - regole pertinenti,
  - vincoli formato output.
- Include parametri consigliati per Minerva 7B fine-tuned e guardrail.

## `data/piano_dataset.md`

- Definisce i volumi target (MVP), la strategia di campionamento e gli split.
- Stabilisce i quality gates minimi da superare batch per batch.
- Elenca metriche per miglioramenti iterativi.

## `data/schema_annotazione.json`

- È il contratto dati del record finale.
- Campi principali:
  - `source_text_it`,
  - `lis_gloss`,
  - `signs` (annotazioni per segno),
  - `references` (entity-locus),
  - `generation_meta` (tracciabilità).
- Ogni record JSONL deve rispettarlo prima del fine-tuning.

## `data/vocabolario.json`

- Lessico seed con parametri articolatori essenziali.
- Ogni lemma ha:
  - configurazione,
  - luogo,
  - movimento,
  - orientamento,
  - tipo grammaticale.
- È la fonte primaria usata dal retriever e dal generator baseline.

## `data/input_frasi.txt`

- Frasi seed italiane (una per riga) usate per generazione batch.
- Serve come dataset di avvio per testare la pipeline.

## `output/dataset_lis_final.jsonl`

- File risultato: un record JSON per riga.
- Pensato per training pipeline-friendly e processing streaming.
