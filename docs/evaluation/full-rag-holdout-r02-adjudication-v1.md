# Full RAG holdout R02 — adjudicação v1

`full-rag-holdout-r02-adjudication-v1` é uma correção versionada da revisão
humana do R02. Ela não substitui nem modifica o resumo operacional R02:
ambos os veredictos permanecem `FULL_RAG_HOLDOUT_FAILED`.

O artefato é intencionalmente sanitizado. Ele contém apenas classificações,
booleans, métricas agregadas e SHA-256 de evidências imutáveis. Não contém
perguntas, respostas, citações, trechos, contexto, prompts, identificadores
operacionais, horários, caminhos absolutos ou segredos.

Os arquivos são:

- `evals/rag/full-rag-holdout-r02-adjudication-v1.json`
- `evals/rag/full-rag-holdout-r02-adjudication-v1.schema.json`
- `scripts/validate_full_rag_holdout_r02_adjudication.py`

O validador verifica o hash fixo do resumo original, os 12 reports R02, a
fixture, o freeze, a política de threshold, o manifesto do índice e o canary
D02. Assim, uma alteração em qualquer evidência invalida a adjudicação em vez
de permitir que sua história seja reescrita.

Execute somente o gate offline:

```powershell
uv run --locked python scripts/validate_full_rag_holdout_r02_adjudication.py
```

As classificações técnicas preservam os failures de grounding em `HOLD-P02`,
`HOLD-P03`, `HOLD-P04` e `HOLD-N04`. `HOLD-P05` é separado como
`EVALUATION_ORACLE_DEFECT`, portanto não é classificado como falha do modelo.
Uma próxima fase diagnóstica deve usar apenas os códigos sanitizados incluídos
no artefato para diferenciar citação duplicada, marcador inválido, divergência
de citação, citação ausente, schema inválido, grounding unsupported e saída
inválida do provider.
