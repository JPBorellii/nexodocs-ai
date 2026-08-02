# ADR-008 — Normalize the OpenAI Structured Outputs schema and contain provider errors

## Contexto

A primeira tentativa do holdout completo, `full-rag-holdout-r01`, terminou antes de avaliar
qualquer caso. O incidente sanitizado registra `TECHNICALLY_FAILED`, decisão de qualidade
`NOT_EVALUATED`, HTTP 400 no `ANSWER_PROVIDER`, stdout JSON ausente e ausência de usage report.
O artefato `evals/rag/full-rag-holdout-r01-technical-incident.json` não contém pergunta,
resposta, traceback, mensagem do provedor, request ID, timestamp ou caminho.

O provider enviava diretamente o contrato canônico local para Structured Outputs. Esse contrato
inclui `$schema` e `uniqueItems`. A causa exata da resposta não foi preservada; a conclusão é,
portanto, uma causa provável fortemente sustentada —
`OPENAI_STRUCTURED_OUTPUT_SCHEMA_INCOMPATIBILITY` — e não uma confirmação absoluta.

## Decisão

O contrato canônico `knowledge_base/metadata/rag-generated-answer.schema.json` permanece
inalterado e continua sendo usado na validação local posterior à resposta. A projeção determinística
`openai_generated_answer_schema()` cria uma cópia profunda, remove somente `$schema` e
`uniqueItems`, e é a única estrutura enviada ao provider OpenAI. Assim, campos, limites,
`additionalProperties: false`, propriedades e `required` são preservados, enquanto citações
duplicadas continuam rejeitadas pelo contrato local e pela validação de grounding.

O provider captura apenas as exceções conhecidas do SDK OpenAI e as converte em `ProviderError`
sanitizado. O pipeline então devolve `generation_failed` com `provider_invalid_output`; a CLI
emite JSON controlado, cria o usage report opt-in privacy-safe e retorna código 1, sem traceback.
Erros de programação continuam visíveis, pois não há captura genérica.

Uma nova fixture, `full-rag-holdout-r02`, preserva as doze queries e expectativas de R01 e registra
R01 como predecessor técnico não avaliado. O system freeze R02 inclui a fronteira corrigida do
provider. Nenhum relatório operacional R02 é criado por esta mudança.

## Consequências e riscos

- Answerability, preflight de segurança, bloqueio clínico, proteção contra prompt injection,
  fallback, grounding e regras de citação não mudam.
- Prompt, versão/hash de prompt, threshold, retrieval, embeddings, documentos, chunks e manifesto
  permanecem congelados.
- A projeção é uma política local de compatibilidade; a disponibilidade e regras finais da API
  devem ser confirmadas somente em execução futura autorizada.
- R02 ainda precisa de preflight e execução real separados; esta decisão não realizou chamadas
  OpenAI, embeddings reais ou Qdrant real.
