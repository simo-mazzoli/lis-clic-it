# lis-clic-it

Progetto per costruire un **dataset LIS (Lingua dei Segni Italiana)** in formato JSONL, generato con un flusso **RAG + LLM** e pensato per il **fine-tuning di Minerva 7B**.

> Nota importante: questo repository non descrive la “lingua LIS completa”, ma un percorso **controllato e scalabile** su subset linguistici ad alta priorità.

## Obiettivo

Produrre un dataset sintetico ma linguisticamente controllato, con:
- frase sorgente in italiano;
- glossa/struttura LIS;
- annotazioni formali del segno (manuali, spaziali, non-manuali, morfosintassi);
- metadata di qualità e tracciabilità della generazione.

## Perimetro linguistico (subset iniziale)

Il progetto si concentra su fenomeni utili per il training iniziale:
- **coreferenza** (referenti nello spazio di segnatura);
- **verbi flessivi/direzionali** (accordo con loci);
- **frasi affermative**;
- **interrogative polari** (yes/no);
- **variazioni di ordine e topicalizzazione** dove pertinente.

## Struttura repository

```text
/workspace/lis-clic-it
├── data/
│   ├── vocabolario.json             # lessico LIS con parametri di realizzazione
│   ├── schema_annotazione.json      # schema dei record dataset (contract dati)
│   ├── regole_grammatica.md         # linee guida linguistiche operative
│   ├── linee_guida_rag.md           # come usare retrieval + prompt Minerva 7B
│   ├── piano_dataset.md             # strategia di campionamento e quality gates
│   └── input_frasi.txt              # frasi seed in italiano
├── scripts/
│   ├── retriever.py                 # (placeholder) recupero lessico/regole
│   └── generator.py                 # (placeholder) generazione JSONL con LLM
├── output/
│   └── dataset_lis_final.jsonl      # output finale per training
└── README.md
```

## Pipeline consigliata (end-to-end)

1. **Curazione lessicale**
   - Popolare `data/vocabolario.json` con segni validati.
   - Ogni lemma deve includere forma base + varianti + vincoli d’uso.

2. **Formalizzazione schema**
   - Usare `data/schema_annotazione.json` come contratto obbligatorio.
   - Validare ogni esempio generato contro lo schema.

3. **RAG linguistico**
   - Recuperare regole da `regole_grammatica.md` e lessico pertinente.
   - Iniettare nel prompt solo il contesto necessario (riduce allucinazioni).

4. **Generazione controllata (Minerva 7B fine-tuned)**
   - Prompt con istruzioni rigide di output JSON.
   - Decoding deterministico o a bassa temperatura.

5. **Validazione automatica + revisione umana**
   - Check schema JSON.
   - Check coerenza loci/coreferenza.
   - Revisione LIS expert su campioni stratificati.

6. **Esportazione JSONL**
   - Un record per riga, pronto per pipeline di fine-tuning.

## Principi di qualità dati

- **Tracciabilità**: ogni record deve avere `id`, `source`, `generation_meta`.
- **Bilanciamento**: distribuzione uniforme per tipo frase/fenomeno.
- **No leakage**: separazione train/dev/test per template e lessico.
- **Versionamento**: incrementare `dataset_version` a ogni release.

## Prossimi step implementativi

- Implementare `scripts/retriever.py` per retrieval su lessico + regole.
- Implementare `scripts/generator.py` per batch generation + validazione schema.
- Aggiungere script di split train/dev/test con controlli anti-duplicato semantico.
- Definire protocollo di valutazione con metriche automatiche + giudizio esperto.

## Avvertenza metodologica

I dati sintetici non sostituiscono la validazione di persone sorde segnanti e linguisti LIS.
La pipeline deve essere progettata come **human-in-the-loop**, non fully automatic.