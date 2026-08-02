# Holdout de recuperação r01

## Resultado preservado

O resultado da tentativa `1` é `HOLDOUT_FAILED`. O artefato versionado
`evals/retrieval/holdout-r01-result.json` é sanitizado e determinístico: não contém perguntas,
textos de chunks, vetores, respostas, caminhos, URLs, timestamps, payloads ou credenciais. O schema
Draft 2020-12 fechado está em `evals/retrieval/holdout-result.schema.json` e o validador offline é
`scripts/validate_retrieval_holdout_result.py`.

| Medida | Resultado |
| --- | ---: |
| Casos supported | 6/6 |
| Recall@5 e Hit Rate@5 | 1.0 |
| Top-1 | 5/6 |
| MRR | 0.91666667 |
| Fallback unsupported | 3/6 |
| Status correto | 9/12 |

O artefato vincula hashes da política congelada, manifesto do índice, chunks, manifesto processado
e plano de índice. Ele não depende de `data/run-reports`, `data/qdrant`, internet, OpenAI, Qdrant
ou variáveis sensíveis.

## Limite do threshold

O menor score do documento esperado para um caso supported é `0.52130791` (HOLD-P04). O maior
score de um falso positivo unsupported é `0.52337769` (HOLD-N04). O gap recalculado é
`-0.00206978`: preservar todos os supported requer threshold menor ou igual ao floor, enquanto
rejeitar o falso positivo requer threshold maior que o ceiling. Portanto,
`threshold_only_separation_possible` é `false`.

`retrieval-threshold-v1` continua congelada em `0.46`; ela não é recalibrada nem invalidada
retroativamente por este resultado. A falha demonstra que relevância semântica não prova que a
evidência responde à solicitação.

## Inspeção sanitizada

Os falsos positivos retornaram, respectivamente, a política de pessoas, a tabela de coberturas e o
diretório organizacional. Isso indica sobreposição semântica de domínio ou de contexto
organizacional, sem prova de answerability. No caso limiting supported HOLD-P04, o documento de
privacidade esperado foi encontrado na posição 2, abaixo de um documento operacional relacionado.

## Próxima etapa

O próximo teste é `full_rag_holdout_r01`, executado contra o sistema RAG completo sem alterações
prévias. Até essa avaliação, não modificar prompt, geração, fallback, schema RAG, retrieval runtime
ou threshold. Qualquer correção posterior exige nova tentativa identificada e evidência separada.

O preflight usa a fixture fixa `evals/rag/full-rag-holdout-r01-cases.json`, seu schema Draft 2020-12
e `scripts/validate_full_rag_holdout_fixture.py`. A tentativa continua sendo baseline: não há gate
semântico de answerability nem regras especiais para salário, Wi-Fi ou faturamento.
