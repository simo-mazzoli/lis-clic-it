# Regole grammaticali operative LIS (subset iniziale)

Questo documento definisce regole *operative* per la generazione del dataset.
Non è una grammatica esaustiva della LIS.

## 1) Riferimento spaziale e coreferenza

1. Quando un referente viene introdotto, assegnare un **locus** (`A`, `B`, `C`...).
2. Le riprese anaforiche devono usare lo stesso locus.
3. Se un referente cambia ruolo discorsivo, esplicitare il cambio nei metadati.

### Esempio
- IT: "Maria vede Luca. Poi lo saluta."
- LIS (astratto): `MARIA[A] LUCA[B] VEDERE(A->B). POI SALUTARE(A->B).`

## 2) Verbi flessivi/direzionali

1. I verbi direzionali devono codificare direzione `source_locus -> target_locus`.
2. Se la frase italiana è ambigua, preferire l'interpretazione esplicita nel campo `notes`.
3. Se il verbo non è flessivo nel lessico, usare forma neutra.

## 3) Frasi affermative

1. Struttura base preferita: `TOPIC (opzionale) + COMMENT`.
2. Mantenere marcatori non manuali se semanticamente rilevanti.
3. Evitare aggiunte lessicali non presenti o non inferibili dal contesto.

## 4) Interrogative polari (yes/no)

1. Marcare il tipo frase con `sentence_type = "polar_question"`.
2. Includere annotazione non-manuale (es. sopracciglia alzate) in `non_manuals`.
3. Non introdurre wh-signs se non richiesti.

## 5) Topicalizzazione (quando presente)

1. Se un costituente è topicalizzato, segnalarlo con `topic = true`.
2. Evitare topicalizzazione multipla nelle frasi seed iniziali.

## 6) Normalizzazione glosse

1. Gloss in maiuscolo (`MELA`, `DARE`, `VEDERE`).
2. Token separati da spazio, punteggiatura minima.
3. Informazioni morfologiche in attributi strutturati, non nel token grezzo.

## 7) Policy anti-allucinazione

1. Ogni glossa deve essere riconducibile a lemma presente in `vocabolario.json` o in una lista OOV esplicita.
2. Se manca il lemma, valorizzare `oov=true` e `confidence` bassa.
3. Non inventare parametri articolatori non supportati.
