# ADR-010 — Exigir sinais sanitizados antes de alterar a validação de quotes

## Status

Aceito.

## Contexto

O D03 reproduziu `grounding_quote_not_in_evidence` em `HOLD-P02` e `HOLD-P04`, enquanto
`HOLD-P03` e `HOLD-N04` não reproduziram a falha. O código atual faz uma comparação de substring
literal, case-sensitive e sem canonicalização. Testes sintéticos provam rejeições causadas apenas
por diferenças de whitespace, Unicode, pontuação e dash.

O código sanitizado do D03, isoladamente, não informa se P02/P04 falharam por canonicalização,
paráfrase, quote gerada incorretamente, associação ao citation_id errado ou diferença estrutural da
evidência. Atribuir uma dessas causas aos casos reais seria extrapolar a evidência preservada.

## Decisão

Classificar a causa como `ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04`. Não alterar runtime nesta fase. Um
D04 futuro, explicitamente autorizado, poderá observar somente:

- correspondência exata, casefold, whitespace normalizado, NFC, NFKC, pontuação e dash;
- buckets de comprimento da quote e da evidência;
- associação ao mesmo citation_id e existência em outro citation_id;
- buckets de sobreposição de tokens e caracteres;
- estágio do validador e safe error code.

O contrato conceitual fica em
`evals/rag/grounding-diagnostic-d04-signals-concept.schema.json`. É proibido persistir quote,
evidência, resposta, pergunta, citação, excerpt, tokens textuais, hashes derivados de conteúdo,
documentos, chunks, contexto, prompt ou request ID. O schema não constitui execução nem artefato de
resultado D04.

## Fase futura

Uma implementação D04, se autorizada, provavelmente tocaria
`src/nexodocs_ai/rag/validator.py`, `src/nexodocs_ai/rag/grounding_diagnostics.py`,
`src/nexodocs_ai/rag/pipeline.py` e a fronteira fechada de observabilidade. Ela deverá ser coberta
por testes unitários de cada sinal, testes de privacidade, testes de integração do fallback e prova
de que nenhum conteúdo ou identificador atravessa a fronteira sanitizada.

Somente os sinais D04 poderão adjudicar entre canonicalização, drift do modelo, associação errada
ou outra causa. Uma eventual mudança de canonicalização exigirá contrato explícito sobre quais
transformações preservam significado e testes contra falsos positivos.

## Consequências

R02 permanece `FULL_RAG_HOLDOUT_FAILED`; D03 permanece `NOT_APPLICABLE` para decisão de qualidade;
P03 e N04 permanecem `NON_REPRODUCED_IN_D03`; R03 não existe. Nenhum runtime, prompt, threshold,
retrieval, grounding ou observabilidade foi alterado nesta decisão.
