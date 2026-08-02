# ADR-006 — Pré-registro do threshold de recuperação semântica

## Contexto

A calibração controlada da recuperação semântica separou casos com suporte dos sem suporte antes da validação independente. O índice materializado contém 44 pontos para `text-embedding-3-small`, 1536 dimensões, distância Cosine e coleção `nexodocs_chunks_v1`. O threshold não pode ser ajustado após observar o holdout.

## Decisão

O repositório pré-registra a política `retrieval-threshold-v1` em `knowledge_base/index/retrieval-threshold-policy.json`, validada pelo schema fechado `retrieval-threshold-policy.schema.json` e pelo validador offline correspondente.

O positive floor é `0.56656004` e o negative ceiling é `0.35101858`. O midpoint calculado é `0.45878931`, cujo candidato arredondado a seis casas é `0.458789`. O valor operacional congelado é `0.46`, com comparação `greater_than_or_equal`, após arredondamento a duas casas. As margens resultantes são `0.10656004` e `0.10898142`, e o gap é `0.21554146`.

`RETRIEVAL_SCORE_THRESHOLD=0.46` deve ser definido explicitamente no ambiente de execução. A política não é um default universal: ela é vinculada aos hashes de chunks, manifesto processado, plano de índice, manifesto do índice e resumo sanitizado da calibração.

## Consequências

- O holdout independente só pode validar a política; não pode editar este artefato.
- Falha no holdout exige uma nova versão de política e nova calibração, nunca alteração retroativa.
- Alterar modelo, dimensões, chunking, coleção ou distância exige nova calibração e nova política.
- A validação é offline e não usa OpenAI, Qdrant ativo, `data/qdrant` ou `data/run-reports`.
- A política preserva somente metadados sanitizados e determinísticos; não contém perguntas, textos, vetores, prompts, respostas, URLs, paths, timestamps ou credenciais.
