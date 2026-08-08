# ADR-011 — Fronteira opt-in sanitizada para o grounding diagnostic D04

## Status

Aceito.

## Contexto

O D03 reproduziu `grounding_quote_not_in_evidence` somente em `HOLD-P02` e `HOLD-P04`. O
validador usa substring Python literal, sensível a caixa, exclusivamente no `citation_id`
declarado. O resultado D03 é `NOT_APPLICABLE`, a causa permanece
`ROOT_CAUSE_INSUFFICIENT_REQUIRE_D04` e R03 não existe.

Diagnosticar a causa exige comparar temporariamente quote e evidência, mas esses textos não podem
atravessar a fronteira persistente. As normalizações também não podem flexibilizar o grounding.

## Decisão

O validador e sua ordem de predicados permanecem intocados. Quando a CLI recebe simultaneamente
`--sanitized-grounding-diagnostic-report`, o case ID permitido, um usage report e o modo
privacy-safe, o pipeline calcula a projeção somente depois de o validador rejeitar a resposta com
`grounding_quote_not_in_evidence`.

A projeção ocorre em memória a partir dos objetos já presentes no processo e retorna somente
booleans e enums fechadas. Quote, evidência e IDs de outro bloco não entram em exceções, logs,
usage report ou artefato. Os signal sets são ordenados pela projeção sanitizada, não pela posição
da citação. Entradas de normalização são limitadas a 16.000 caracteres; truncamento torna-se um
boolean e conduz a `INCONCLUSIVE` quando não há sinal mais forte.

O artefato é publicado completo por hard link exclusivo a partir de um temporário criado com
semântica `O_EXCL`. Destino preexistente nunca é sobrescrito e nenhum parcial permanece. O caminho
fica restrito a `data/run-reports` e não é persistido no JSON.

Um primeiro preflight independente bloqueou a execução ao demonstrar colisão entre caminhos de
usage/D04, escape de falhas operacionais e validação incompleta das fontes. A fronteira passa a
comparar destinos resolvidos, normalizados e sem distinção de caixa, usando `samefile` quando ambos
existem, antes da construção de providers. Falhas de mkdir, temporário, descriptor, fsync,
publicação ou cleanup são convertidas em erro D04 estático com contexto de sistema suprimido.

Validação estrutural de CI requer `--schema-only`. O modo operacional padrão do artefato exige seu
usage report; o consolidado exige os artefatos P02 e P04. Nenhuma mensagem de sucesso operacional
pode ser emitida sem confrontar as fontes correspondentes.

## Consequências

Sem a flag, o pipeline não calcula nem transporta sinais. Com a flag, status, safe error, exit code,
busca, geração, retries, calls e tokens continuam os mesmos; D04 apenas observa a rejeição. A
instrumentação não aprova resposta, não emite `PASSED`/`FAILED`, mantém a decisão
`NOT_APPLICABLE` e registra `r03_created: false`.

Há risco residual de sinais amplos não distinguirem paráfrase de geração incorreta; por isso as
classes usam apenas graus de sobreposição ou `INCONCLUSIVE`, nunca atribuição de culpa.
