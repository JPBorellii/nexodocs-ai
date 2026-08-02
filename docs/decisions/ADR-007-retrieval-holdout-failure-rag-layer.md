# ADR-007 — Record retrieval holdout failure and evaluate answerability at the RAG layer

## Contexto

O holdout independente `retrieval-holdout-r01`, tentativa 1, foi executado com
`retrieval-threshold-v1` congelada em `0.46`, sobre o índice de 44 pontos de
`text-embedding-3-small`, 1536 dimensões, Cosine e `nexodocs_chunks_v1`. O resultado sanitizado
versionado registra seis casos supported e seis unsupported, sem geração RAG.

Os casos supported foram recuperados (Recall@5 e Hit Rate@5 de `1.0`; Top-1 de `0.83333333`; MRR de
`0.91666667`). O fallback unsupported atingiu `0.5`: três casos semanticamente próximos receberam
candidatos, embora esses candidatos não comprovassem answerability. O status correto total foi
`0.75`.

O menor score do documento esperado supported é `0.52130791`; o maior falso positivo unsupported é
`0.52337769`. O gap é `-0.00206978`. Preservar todos os supported exigiria threshold menor ou igual
ao primeiro valor; rejeitar esse falso positivo exigiria threshold maior que o segundo. As condições
são incompatíveis.

## Decisão

Registrar permanentemente `HOLDOUT_FAILED` em artefato sanitizado, schema fechado, validador offline
e testes. `retrieval-threshold-v1` permanece válida e congelada como a política candidata da etapa
de retrieval; esta decisão não a altera, renomeia, recalibra ou invalida retroativamente.

Não tratar relevância semântica como resposta suportada. A próxima avaliação será
`full_rag_holdout_r01`, cobrindo recuperação, suficiência de evidência, geração fundamentada,
citações, fallback `no_evidence`, proteções clínicas e resistência a prompt injection. Antes desse
holdout, não modificar prompt, geração, fallback, schema RAG, retrieval runtime ou threshold.

## Consequências e riscos

- A recuperação de casos supported está aprovada, mas o resultado global permanece falho.
- A camada RAG pode rejeitar evidência insuficiente mesmo quando retrieval retorna candidatos; suas
  citações devem permanecer vinculadas aos chunks de evidência retornados.
- O risco atual é aceitar conteúdo relacionado que não responde à solicitação. O holdout RAG mede
  esse risco no fluxo completo, sem mascará-lo por uma alteração antecipada.
- Se o holdout RAG falhar, uma tarefa posterior define a correção e uma nova tentativa identificada.

## Alternativas rejeitadas

1. Aumentar o threshold até eliminar os negativos: perderia o caso supported limitante.
2. Reduzir, trocar ou reclassificar perguntas do holdout: alteraria a evidência observada.
3. Marcar o holdout como sucesso parcial: ocultaria a falha do fallback.
4. Apagar evidências locais: destruiria a auditabilidade do resultado.
5. Alterar prompt ou RAG antes da avaliação completa: contaminaria a medição.
6. Adicionar regras específicas para exemplos individuais: introduziria overfitting frágil.
7. Excluir resultados semanticamente próximos com palavras-chave: não resolveria answerability de
   forma geral.
