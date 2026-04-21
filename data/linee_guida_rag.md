# Linee guida RAG per generazione dataset LIS con Minerva 7B

## Obiettivo
Usare retrieval mirato per fornire al modello solo:
1) lessico necessario,
2) regole grammaticali pertinenti,
3) vincoli di output JSON.

## Contesto da recuperare per ogni frase

Per una frase italiana in input:
- estrarre lemmi candidati;
- recuperare voci da `vocabolario.json`;
- recuperare sezioni regole da `regole_grammatica.md` in base al tipo frase;
- costruire prompt con 3 blocchi: `LESSICO`, `REGOLE`, `TASK`.

## Prompt template (sintesi)

- System: "Sei un annotatore LIS. Rispondi solo JSON valido." 
- Context:
  - lessico rilevante;
  - regole subset (coreferenza, verbo flessivo, interrogativa, ecc.);
  - schema record sintetico.
- User task:
  - frase italiana;
  - output richiesto in 1 record JSON con campi obbligatori.

## Parametri consigliati Minerva 7B (fine-tuned)

- temperatura: 0.1 - 0.3
- top_p: 0.9
- max_new_tokens: sufficiente al record completo
- stop sequence: fine oggetto JSON

## Guardrail

1. Rifiutare output non JSON.
2. Validare contro `schema_annotazione.json`.
3. Se confidence < soglia, inviare in coda revisione umana.

## Tracciabilità

Salvare per ogni esempio:
- hash prompt,
- versione lessico,
- versione regole,
- timestamp generazione,
- modello e checkpoint.
