# Piano dataset per fine-tuning

## Target iniziale (MVP)

- 10k - 30k record JSONL.
- Distribuzione bilanciata tra:
  - affermative,
  - interrogative polari,
  - coreferenza semplice,
  - verbi flessivi con 2 referenti.

## Strategia di campionamento

1. Template linguistici controllati.
2. Variazione lessicale guidata da classi semantiche.
3. Aumento graduale complessità (curriculum):
   - Fase 1: frasi brevi mono-proposizionali.
   - Fase 2: coreferenza intra-frase.
   - Fase 3: combinazione coreferenza + verbo flessivo + interrogativa.

## Split dati

- Train: 80%
- Dev: 10%
- Test: 10%

Vincoli:
- evitare template identici in train e test;
- limitare sovrapposizione lessicale ad alta frequenza nel test;
- creare test set "challenge" con strutture rare.

## Quality gates

Ogni batch deve superare:
1. validazione schema = 100%;
2. tasso OOV sotto soglia definita;
3. coerenza loci/coreferenza su regole deterministiche;
4. revisione umana su campione (es. 5-10%).

## Metriche per iterazioni successive

- JSON validity rate
- grammatical consistency rate
- expert acceptance rate
- errore su fenomeni target (coreferenza/verbi flessivi)
