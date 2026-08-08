# Grounding diagnostic D04 — contrato sanitizado

## Estado e finalidade

O D03 versionado concluiu que `HOLD-P02` e `HOLD-P04` reproduzem
`grounding_quote_not_in_evidence`; P03 e N04 não entram no D04. A execução real D04 foi concluída
no commit `9e1a327d19ca10f0c4757de00c3bb2dfa5887fd9` e seu resultado sanitizado foi preservado sem nova
execução nesta fase. R02 permanece `FULL_RAG_HOLDOUT_FAILED`, a decisão D04 é sempre
`NOT_APPLICABLE` e R03 permanece ausente.

## Resultado real preservado

`HOLD-P02` apresentou duas falhas de quote e `HOLD-P04`, uma. Os dois casos foram classificados
como `WHITESPACE_ONLY_DIFFERENCE`; não houve caso inconclusivo. A classificação final da
investigação é `ROOT_CAUSE_CONFIRMED_WHITESPACE_CANONICALIZATION_GAP`.

O resultado versionado está em `evals/rag/grounding-diagnostic-d04-result-v1.json`. Ele registra
somente identificadores fechados, enums, booleanos, contagens e hashes de integridade. Não contém
query, answer, quote, evidence, texto de citação, documento, chunk, prompt, conteúdo normalizado ou
hashes de quote/evidence. O schema específico está em
`evals/rag/grounding-diagnostic-d04-result-v1.schema.json`.

## Ativação

O modo normal não cria diagnóstico. A ativação exige todos os argumentos abaixo:

```text
--usage-report data/run-reports/<novo-usage>.json
--privacy-safe-usage-report
--sanitized-grounding-diagnostic-report data/run-reports/<novo-d04>.json
--sanitized-grounding-diagnostic-case-id HOLD-P02|HOLD-P04
```

Não há overwrite D04 nem variável global de ativação. Destino inválido ou existente falha com
payload sanitizado, exit code 1 e stderr vazio. Outros status e outros erros de grounding não
produzem D04.

O primeiro preflight independente bloqueou a execução por quatro lacunas: destinos equivalentes
para usage e D04, exceções operacionais anteriores à captura, validação por caso sem usage
obrigatório e consolidação sem os dois artefatos. O hardening posterior passou a rejeitar caminhos
absolutos normalizados equivalentes, incluindo comparação sem distinção de caixa e `samefile`
seguro, antes de construir providers. Um preflight posterior autorizou a execução real já concluída;
esta fase de preservação não a repete.

## Nomes canônicos preservados

Os nomes são contratos documentais; a CLI continua exigindo caminhos explícitos:

```text
data/run-reports/phase-6a-grounding-diagnostic-d04-p02-usage.json
data/run-reports/phase-6a-grounding-diagnostic-d04-p02.json
data/run-reports/phase-6a-grounding-diagnostic-d04-p04-usage.json
data/run-reports/phase-6a-grounding-diagnostic-d04-p04.json
data/run-reports/phase-6a-grounding-diagnostic-d04-summary.json
```

## Sinais persistidos por falha de quote

- matches: `exact_match`, `casefold_match`, `whitespace_normalized_match`, `unicode_nfc_match`,
  `unicode_nfkc_match`, `punctuation_normalized_match`, `quote_and_dash_normalized_match` e
  `combined_normalized_match`;
- associação: `same_citation_id`, `exists_in_other_citation_id` e
  `normalized_match_in_other_citation_id`;
- buckets: `quote_length_bucket`, `evidence_length_bucket`, `character_overlap_bucket` e
  `token_overlap_bucket`;
- caracteres estruturais, somente booleans: NBSP, zero-width, line break, aspas tipográficas e
  dash não ASCII, separadamente para quote e evidência;
- limite: `normalization_input_truncated`;
- constantes: `validator_stage=quote_membership` e
  `original_safe_error_code=grounding_quote_not_in_evidence`.

Não são persistidos texto, contagem exata de caracteres compartilhados, percentual, distância,
tokens textuais, n-grams, posição, outro citation ID ou hashes dos textos.

## Normalizações somente diagnósticas

- `casefold`: `str.casefold`, sem outra transformação;
- whitespace: zero-width vira separador, Unicode whitespace é compactado para um espaço e bordas
  são removidas;
- NFC e NFKC: `unicodedata.normalize` isoladamente;
- tipografia: aspas/apóstrofos curvos para ASCII, en/em e demais dashes Unicode para `-`, reticência
  Unicode para `...`;
- pontuação: remoção de code points cuja categoria Unicode inicia por `P`, sem mudar whitespace;
- combinada: NFKC, casefold, tipografia, remoção de pontuação e compactação de whitespace.

Nenhuma dessas funções participa de `validate_generated` ou aprova uma citação.

## Buckets fechados

Comprimento: `EMPTY` (0), `VERY_SHORT` (1–25), `SHORT` (26–100), `MEDIUM` (101–250),
`LONG` (251–500) e `OVER_LIMIT` (acima de 500). `quote_failure_count` é inteiro de 1 a 16 porque
revela apenas o pequeno número de referências inválidas já produzidas numa resposta, nunca texto.

Sobreposição: `NONE` (0), `LOW` (maior que 0 e menor que 0,34), `MEDIUM` (0,34 a menor que 0,67),
`HIGH` (0,67 a menor que 1) e `EXACT` (1). O cálculo em memória usa Dice sobre multiconjuntos de
caracteres e Jaccard sobre conjuntos de tokens após normalização combinada; somente o bucket sai do
processo.

## Classificação determinística

As classes fechadas são: `EXACT_MATCH_UNEXPECTED`, `CASE_ONLY_DIFFERENCE`,
`WHITESPACE_ONLY_DIFFERENCE`, `UNICODE_NORMALIZATION_DIFFERENCE`, `TYPOGRAPHIC_DIFFERENCE`,
`PUNCTUATION_ONLY_DIFFERENCE`, `WRONG_CITATION_ASSOCIATION`,
`NORMALIZED_MATCH_IN_OTHER_CITATION`, `HIGH_OVERLAP_NON_LITERAL`,
`MEDIUM_OVERLAP_NON_LITERAL`, `LOW_OVERLAP_NON_LITERAL`, `NO_MEANINGFUL_OVERLAP`,
`MULTIPLE_POSSIBLE_CAUSES` e `INCONCLUSIVE`.

A precedência é: match literal inesperado; match literal em outro ID; match normalizado somente em
outro ID; diferença isolada (com preferência explícita para whitespace estrutural e tipografia);
múltiplas normalizações; truncamento/entrada vazia; e, por fim, maior bucket de sobreposição. Mais
de uma classificação entre signal sets consolida como `MULTIPLE_POSSIBLE_CAUSES`.

## Artefatos e validação

O schema Draft 2020-12 por caso é `evals/rag/grounding-diagnostic-d04.schema.json`. Ele contém
identidade/versionamento, commit, hashes das fontes históricas autorizadas, case/status/safe error,
classe, contagem e signal sets, calls/attempts/tokens, hash do usage report, flags de execução e
privacidade, `NOT_APPLICABLE` e `r03_created: false`.

O schema consolidado operacional é `evals/rag/grounding-diagnostic-d04-result.schema.json`:
exatamente os dois casos, counts/classes, classificados/inconclusivos, hashes dos artefatos e
invariantes de integridade/privacidade. O schema `result-v1` preserva adicionalmente commit, hashes
dos usage reports e summary, contagens de falhas de quote e classificação causal final. Os
validadores offline são:

```powershell
uv run --locked python scripts/validate_grounding_diagnostic_d04.py --schema-only
uv run --locked python scripts/validate_grounding_diagnostic_d04_result.py
uv run --locked python scripts/validate_grounding_diagnostic_d04_result.py --verify-local-sources
```

O modo padrão do resultado é CI-safe e não exige os relatórios ignorados. O modo opcional
`--verify-local-sources` verifica por hash os dois usage reports, os dois artefatos D04 e o summary,
além de validar schema, privacidade e coerência, sem imprimir conteúdo. Para um artefato isolado, o
modo padrão exige `--artifact` e `--usage-report`; o modo operacional legado do summary exige
`--result`, `--p02-artifact` e `--p04-artifact`. Combinar modos incompatíveis é recusado.

Falhas de mkdir, temporário, descriptor, flush, fsync, hard link e cleanup são contidas na fronteira
D04. O payload público contém somente status e código estático, nunca caminho ou texto do sistema;
stderr permanece vazio. A publicação usa hard link exclusivo e cleanup best-effort com retry.

## Preservação concluída — não reexecutar nesta fase

Esta fase não executa providers, retrieval ou perguntas e não altera os relatórios reais em
`data/run-reports`. A próxima fase é a correção mínima de canonicalização de whitespace no
grounding, mantendo evidências, fallback e testes de regressão.

Riscos residuais: buckets perdem detalhe deliberadamente; match normalizado não prova equivalência
semântica; e sobreposição alta não distingue paráfrase de texto gerado. Esses casos permanecem
diagnóstico, não julgamento de qualidade.
