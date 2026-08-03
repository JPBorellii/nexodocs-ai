# Grounding diagnostic D03 — resultado consolidado v1

## Resultado preservado

O D03 foi concluído no commit `d2f0a457d4d80fd56fa57ab54489464a54efbb98`, sem nova
execução nesta fase. Quatro casos foram executados. `HOLD-P02` e `HOLD-P04` reproduziram
`grounding_quote_not_in_evidence`; `HOLD-P03` e `HOLD-N04` permanecem
`NON_REPRODUCED_IN_D03`. Não reprodução não significa correção, estabilidade ou aprovação.

O R02 continua `FULL_RAG_HOLDOUT_FAILED`. O D03 é diagnóstico e não emite decisão de qualidade:
`holdout_quality_decision` permanece `NOT_APPLICABLE`. O threshold `0.46`, o top-k `5`, os
modelos congelados e o histórico R02 foram preservados. R03 não foi criado.

O resultado versionado está em `evals/rag/grounding-diagnostic-d03-result-v1.json`, com SHA-256
`62c54027590b00cf3222e3cedc54f4c6be52db7b5ef9b8bc48d355de8f94a64f`. Ele contém somente
identificadores fechados, enums, booleanos, contagens e hashes de integridade. Nenhuma pergunta,
resposta, quote, citação, evidência, contexto, chunk, prompt, request ID, timestamp ou caminho foi
persistido.

| Caso | Estado D03 | Classificação | Usage SHA-256 | Diagnóstico SHA-256 |
| --- | --- | --- | --- | --- |
| HOLD-P02 | `grounding_failed` | `REPRODUCED_GROUNDING_FAILURE` | `7cc78f487b1d235c178981a58c64a3ba265c9b982902d38e7b94ff1d01f1efdc` | `9cb7a0b2acacbefc4a35a9f220a137a8ac7f630a63bacd48bee3899d2bfea90f` |
| HOLD-P03 | `answered` | `NON_REPRODUCED_IN_D03` | `bd3f30bcb4001f760f07da378dc8289fbaa7599de3d52fbd82d0b148584974d9` | `null` |
| HOLD-P04 | `grounding_failed` | `REPRODUCED_GROUNDING_FAILURE` | `8b763ad18ceba298b2bfd5f721f1ca26f4c33f1ccbf48b51967abea5e5757dfb` | `4fc88fbc90d232e5f3d9277b3380d1b60f076f4cc288f49fc0f63b5ca4698cfc` |
| HOLD-N04 | `answered` | `NON_REPRODUCED_IN_D03` | `64112dcef7b95f4658dd14011fa59a2a6a4ffde9a736c8dec5c59b4beda2812e` | `null` |

Os dois diagnósticos passaram pelo schema e validador D03 existentes. Os quatro usage reports
passaram pelo contrato privacy-safe. Retrieval e answer registraram, em cada caso, uma logical API
call e uma physical attempt; os totais de tokens foram coerentes sempre que todos os componentes
estavam disponíveis.

## Algoritmo atual

Em `src/nexodocs_ai/rag/validator.py`, `validate_generated` recebe a quote em
`GeneratedCitationReference.quote` e a evidência em `EvidenceBlock.text`. Na linha aproximada 38,
os blocos são associados por `evidence_id`; nas linhas 40–57, cada quote é associada somente ao
bloco cujo ID coincide com `citation_id`.

Os predicados de citação são avaliados com short-circuit nesta ordem: citation_id já declarado;
citation_id desconhecido; quote vazia; quote acima de 500 caracteres; e
`item.quote not in available[item.citation_id].text`. A última operação é a busca de substring
literal do Python, não igualdade. Ela é sensível a maiúsculas, espaços, tabs, CR/LF, espaço não
separável, pontos de código Unicode, formas NFC/NFD/NFKC, aspas, apóstrofos, hífens, travessões,
pontuação e caracteres invisíveis. Não há `strip`, casefold ou normalização de whitespace,
Unicode, pontuação, dash ou elipse. Uma quote de espaços não é vazia para `not item.quote`.

Se a substring não pertence à evidência do citation_id declarado, a linha aproximada 55 seleciona
`GroundingErrorCode.QUOTE_NOT_IN_EVIDENCE`; `GroundingValidationError` é lançado na linha 56. Em
`src/nexodocs_ai/rag/pipeline.py`, linhas aproximadas 146–165, a exceção é capturada e
`safe_grounding_error_code` preserva `grounding_quote_not_in_evidence` no fallback sanitizado.
Uma quote existente apenas em outro citation_id é rejeitada. Trechos contíguos truncados no início
ou fim são aceitos porque a operação é substring; paráfrases e remoções internas não são.

Depois das quotes, o runtime valida a presença e o conjunto de markers. O último predicado em
`validator.py`, por volta da linha 85, procura uma resposta longa sem marker depois de uma etapa que
já exigiu markers e os preservou na renderização; na execução normal ele não muda o resultado da
matriz de quotes.

## Matriz sintética

Foram caracterizados 41 cenários totalmente fictícios. Aceitos (15): literal exatamente igual;
substring literal; tab literal; LF literal; CRLF literal; NFC literal; NFD literal; hífen ASCII
literal; truncamento contíguo no início; truncamento contíguo no fim; quote somente de espaços
quando a mesma sequência existe na evidência; quote com exatamente 500 caracteres; carriage return
isolado literal; quote no citation_id correto; e a mesma quote em dois citation_ids distintos cujas
evidências a contêm.

Rejeitados com `grounding_quote_not_in_evidence` (24): diferença apenas de caixa; espaços
duplicados; espaços removidos; quebra substituída por espaço; espaço não separável versus espaço
ASCII; NFC versus NFD; equivalência NFKC; acento combinado versus precomposto; aspas retas versus
curvas; apóstrofo reto versus tipográfico; en dash versus hífen; em dash versus hífen; pontuação
final diferente; vírgula diferente; ponto e vírgula diferente; elipse ASCII versus reticências
Unicode em ambas as direções; palavras removidas; palavras adicionadas; paráfrase semanticamente
equivalente; evidência vazia com quote não vazia; Unicode visualmente semelhante; zero-width space;
e quote encontrada somente em outro citation_id.

Rejeitados com `grounding_quote_mismatch` (2): quote vazia e quote acima de 500 caracteres. Não
houve ajuste de expectativa para ocultar os resultados: a aceitação de whitespace-only e de
truncamentos contíguos documenta exatamente o contrato atual.

## Classificação causal e D04

A matriz prova que diferenças representacionais podem ser rejeitadas. Porém, os artefatos
sanitizados de P02 e P04 preservam somente o código final; eles não distinguem canonicalização,
paráfrase gerada, citation_id incorreto, estrutura da evidência ou outra causa interna. A existência
de uma lacuna no contrato não prova que essa lacuna causou os dois casos reais.

Por isso, a classificação é `ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04`. Um D04 futuro é necessário antes
de escolher uma correção runtime. O schema estritamente conceitual
`evals/rag/grounding-diagnostic-d04-signals-concept.schema.json` limita a observação a booleanos e
buckets sanitizados. Nenhuma execução D04, fixture R03, artefato R03 ou alteração runtime foi criada.
