# lis-clic-it

Questo progetto mira a creare un dataset strutturato di frasi in **Lingua dei Segni Italiana (LIS)** utilizzando un approccio **RAG (Retrieval-Augmented Generation)** e il modello linguistico **Minerva 7B**. L'obiettivo è produrre un file JSONL pronto per il fine-tuning di modelli specializzati nella traduzione verso la LIS.

## 📌 Obiettivi del Progetto
- Generare glosse LIS con annotazioni spaziali e morfosintattiche.
- Focalizzarsi su subset linguistici complessi: coreferenza, verbi flessivi, interrogative polari.
- Superare il limite della scarsità di dati (low-resource) tramite la generazione sintetica controllata.

---

## 📂 Organizzazione del Progetto (File Structure)

Per un'implementazione rapida (1 mese), mantenere la seguente struttura di cartelle:

```text
LIS_Project/
├── data/
│   ├── vocabolario.json          # Database dei segni con parametri manuali
│   ├── regole_grammatica.md      # Linee guida sintattiche per il RAG
│   └── input_frasi.txt           # Frasi in italiano da tradurre (una per riga)
├── scripts/
│   ├── retriever.py              # Logica per estrarre dati dal vocabolario
│   └── generator.py              # Script principale per interrogare Minerva 7B
├── output/
│   └── dataset_lis_final.jsonl   # Dataset generato pronto per il training
└── README.md                     # Documentazione del progetto
