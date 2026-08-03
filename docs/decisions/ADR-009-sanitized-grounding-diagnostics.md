# ADR-009 — Diagnósticos sanitizados de grounding

## Status

Aceito.

## Contexto

O `full-rag-holdout-r02` permaneceu `FULL_RAG_HOLDOUT_FAILED`. A adjudicação confirmou
`grounding_failed` em `HOLD-P02`, `HOLD-P03`, `HOLD-P04` e `HOLD-N04`, mas o código único
`grounding_validation_failed` não distinguia as causas determinísticas já rejeitadas pelo validador.
Mensagens de exceção, respostas, citações e evidências não podem atravessar a fronteira de
observabilidade.

## Decisão

Cada ramo de rejeição existente passa a emitir um membro de uma enumeração fechada:

- `grounding_schema_invalid`;
- `grounding_answer_too_long`;
- `grounding_unsupported_claim`;
- `grounding_duplicate_citation`;
- `grounding_unknown_citation`;
- `grounding_quote_mismatch`;
- `grounding_quote_not_in_evidence`;
- `grounding_missing_citation`;
- `grounding_marker_citation_mismatch`;
- `grounding_validation_failed`, somente para uma `ValidationError` conhecida sem categoria.

O texto da exceção não é contrato. A CLI reduz um `grounding_failed` a `status` e
`safe_error_code`; o relatório privacy-safe conserva apenas metadados permitidos, contagens e uso.
Não há nova tentativa de geração nem segunda chamada. Os predicados, sua ordem, o prompt, o
retrieval, o threshold, o contexto, o schema de resposta, o fallback e os resultados de sucesso
permanecem semanticamente iguais.

O schema `grounding-diagnostic-d03` aceita apenas identificadores de integridade, o caso, o código
sanitizado e contagens de uso. Criar o schema não autoriza nem executa D03.

## Consequências

A causa técnica poderá ser observada em uma execução futura sem persistir conteúdo. Como a
instrumentação pós-R02 altera arquivos cujo conteúdo foi congelado, o freeze R02 continua sendo
validado contra o commit histórico `5db6e1714bd67bbc05000c0fe14b26f06e312f1c`; o artefato de
freeze em si permanece inalterado.

P05 é registrado separadamente como `EVALUATION_ORACLE_DEFECT`. A correção usa o escopo explícito
`R02_RETRIEVED_EVIDENCE`, pois descreve a evidência observada no R02, e não reescreve o documento
canônico. Uma futura fixture R03 deverá preservar a pergunta e usar a expectativa corrigida, mas
nenhum artefato R03 é criado nesta decisão.
