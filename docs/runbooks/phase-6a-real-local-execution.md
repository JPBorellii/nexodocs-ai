# Runbook — Fase 6A: execução real local controlada

## Escopo e condições

Este runbook prepara uma futura execução manual. Não contém chave e não autoriza Qdrant remoto,
prune ou recriação destrutiva. Antes de executar, valide na documentação oficial e na conta a
disponibilidade de `text-embedding-3-small` com 1536 dimensões e `gpt-5.6-luna` com Responses API e
Structured Outputs estrito.

Critérios prévios: `main` sincronizada, worktree limpo, 21 gates aprovados, 44 chunks, 44 pontos,
`data/qdrant` conhecido, `index-manifest.json` validado, política de threshold validada e orçamento
humano aprovado. O threshold semântico já está congelado antes dos smoke tests.

## Configuração não sensível

Em uma sessão PowerShell iniciada na raiz do projeto:

```powershell
$env:APP_ENV = "development"
$env:EMBEDDING_PROVIDER = "openai"
$env:OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
$env:OPENAI_EMBEDDING_DIMENSIONS = "1536"
$env:OPENAI_MAX_RETRIES = "0"
$env:ANSWER_PROVIDER = "openai"
$env:OPENAI_ANSWER_MODEL = "gpt-5.6-luna"
$env:OPENAI_ANSWER_MAX_RETRIES = "0"
$env:OPENAI_ANSWER_MAX_OUTPUT_TOKENS = "1200"
$env:QDRANT_MODE = "local"
$env:QDRANT_PATH = "data/qdrant"
$env:QDRANT_COLLECTION_NAME = "nexodocs_chunks_v1"
$env:RETRIEVAL_SCORE_THRESHOLD = "0.46"
Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue
Remove-Item Env:QDRANT_API_KEY -ErrorAction SilentlyContinue
```

`RAG_MAX_ANSWER_CHARACTERS` limita a resposta pública. Ele não controla tokens da API. O teto de
tokens não é aceito por argumento de pergunta nem por entrada do usuário final.

## Threshold de recuperação congelado

Antes do holdout, valide `knowledge_base/index/retrieval-threshold-policy.json` com
`uv run --locked python scripts/validate_retrieval_threshold_policy.py`. O threshold operacional é
`0.46` e deve ser definido explicitamente por `RETRIEVAL_SCORE_THRESHOLD` no ambiente de execução.
Ele foi calculado a partir do midpoint entre o positive floor `0.56656004` e o negative ceiling
`0.35101858`, com candidato `0.458789` e arredondamento operacional para duas casas.

Esta política está vinculada a `text-embedding-3-small`, 1536 dimensões, Cosine,
`nexodocs_chunks_v1` e ao manifesto congelado. Qualquer troca de modelo, dimensão, chunking ou
coleção exige nova calibração e uma nova versão de política. O holdout não pode alterar este
artefato retroativamente.

## Resultado do holdout retrieval r01

O resultado real já concluído é `HOLDOUT_FAILED`. Os casos supported passaram (Recall@5 e Hit Rate@5
de `1.0`), enquanto o fallback de casos unsupported falhou (`0.5`). O threshold `0.46` permanece
congelado em `retrieval-threshold-v1`: o floor supported de `0.52130791` é inferior ao ceiling
unsupported de `0.52337769`, logo um threshold escalar não separa todos os casos observados.

O resultado foi preservado sem relatórios brutos em `evals/retrieval/holdout-r01-result.json`.
Execute `uv run --locked python scripts/validate_retrieval_holdout_result.py` como o 16º quality
gate. A próxima avaliação é `full_rag_holdout_r01`; não altere threshold, prompt, geração, fallback
ou contratos RAG antes dela.

## Chave

Cole a chave somente em um prompt mascarado do PowerShell local, nunca em comando, arquivo,
relatório, issue, PR ou conversa com o Codex:

```powershell
$secureKey = Read-Host "Cole a OPENAI_API_KEY" -AsSecureString
$keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
try {
    [Environment]::SetEnvironmentVariable(
        "OPENAI_API_KEY",
        [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer),
        "Process"
    )
}
finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    Remove-Variable secureKey, keyPointer -ErrorAction SilentlyContinue
}
```

Confirme apenas presença, sem tamanho ou valor. Ao terminar ou ao primeiro critério de parada:

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_API_KEY", $null, "Process")
Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
```

## Comandos previstos

Use nomes de relatório novos por execução. A sobrescrita exige a opção explícita correspondente.

```powershell
uv run --locked python scripts/index_knowledge_base.py write --usage-report data/run-reports/index-first.json
uv run --locked python scripts/index_knowledge_base.py check-index --usage-report data/run-reports/index-check.json
uv run --locked python scripts/index_knowledge_base.py write --usage-report data/run-reports/index-second.json

uv run --locked python scripts/search_knowledge_base.py "pergunta fictícia" --usage-report data/run-reports/search-01.json
uv run --locked python scripts/answer_question.py --query "pergunta fictícia" --json --threshold 0.46 --usage-report data/run-reports/answer-01.json --privacy-safe-usage-report
```

O primeiro índice vazio deve registrar 44 inseridos e dois lotes lógicos com batch 32. A segunda
indexação deve registrar 44 reutilizados e zero chamadas de embedding. Busca registra apenas uso
do embedding e quantidade de caracteres da pergunta. Resposta separa `retrieval_usage` de
`answer_usage`; bloqueio de preflight registra zero em ambos.

## Conteúdo e segurança do relatório

São permitidos contagens, tokens, status, duração, timestamp UTC, modelos, coleção, dimensão e
request IDs opacos. São proibidos chave, headers, URL, caminho absoluto, variáveis de ambiente,
prompt, contexto, pergunta integral, resposta integral, chunks, payloads e vetores. Não salve
stdout bruto como relatório operacional.

`data/run-reports` e `data/qdrant` permanecem ignorados. O relatório não participa dos artefatos
determinísticos. O `index-manifest.json` futuro não recebe tokens, timestamp, request ID ou duração.

## Chamadas, retries e custos

`logical_api_calls` conta solicitações da aplicação. Com retries de transporte iguais a zero,
`physical_attempts` é auditável e deve ser igual às chamadas lógicas. `application_attempts` conta
separadamente tentativas de geração/validação RAG. Com retries automáticos futuros,
`physical_attempts` pode ser `null`.

Não há preço padrão nem moeda assumida. Após consultar preços oficiais vigentes, calcule sem
arredondamento oculto:

```text
custo_embedding = prompt_tokens_embedding / unidade_do_preço × preço_embedding
custo_resposta = input_tokens / unidade_do_preço × preço_input
               + cached_input_tokens / unidade_do_preço × preço_cached
               + output_tokens / unidade_do_preço × preço_output
```

Valor `null` significa indisponível e não deve ser convertido em zero para cálculo financeiro.

## Critérios de parada

Pare e remova a chave se houver segredo exibido, modo remoto, coleção ou dimensão incompatível,
recriação destrutiva, gate falhando, modelo indisponível, Structured Outputs indisponível, chamada
ou token acima do orçamento, retry inesperado, reembedding na segunda execução, citação inválida,
relatório rejeitado, arquivo Git inesperado ou mudança nos chunks.

## Rollback

Remova primeiro a chave. Não use `git reset --hard`, `git clean` ou prune. Em indexação parcial,
preserve a coleção e repita `check-index`; se compatível, `write` completa apenas pendências. Para
coleção incompatível, preserve o diretório e use nova coleção somente após decisão humana.

Para remover artefatos locais, valide os alvos exatos antes de qualquer exclusão: somente
`data/qdrant`, somente relatórios escolhidos em `data/run-reports` e somente
`knowledge_base/index/index-manifest.json` quando confirmado como não versionado. Ao final, execute
`git status --short`, revise o diff e faça varredura de padrões de segredo sem imprimir conteúdos.

## Grounding diagnostic D03 — preparação, não execução

O R02 permanece `FULL_RAG_HOLDOUT_FAILED` após adjudicação. Os casos `HOLD-P02`, `HOLD-P03`,
`HOLD-P04` e `HOLD-N04` terminaram em `grounding_failed`; o antigo
`grounding_validation_failed` era insuficiente para diagnóstico. A taxonomia sanitizada da
ADR-009 adiciona observabilidade, sem flexibilizar validação nem alterar prompt, provider, limites,
retrieval, threshold, contexto, answerability, preflight, fallback ou número de chamadas.

Antes de uma futura execução isolada, rode os gates offline:

```powershell
uv run --locked python scripts/validate_grounding_error_codes.py
uv run --locked python scripts/validate_grounding_diagnostic.py
uv run --locked python scripts/validate_evaluation_oracle_corrections.py
```

O schema D03 fica em `evals/rag/grounding-diagnostic-d03.schema.json`. Não existe artefato de
execução D03 nesta fase. Um report de `grounding_failed` deve ser criado com
`--privacy-safe-usage-report`; ele não pode conter pergunta, resposta, citação, quote, evidência,
prompt, contexto, traceback, request ID, timestamp ou caminho. `status`, `safe_error_code`, modelos,
contagens de chamadas/tentativas, tokens e `query_character_count` são permitidos.

A correção versionada de P05 fica em `evals/rag/evaluation-oracle-corrections-v1.json`. Ela registra
o defeito do oráculo no escopo da evidência recuperada no R02 e instrui uma futura R03 a preservar a
pergunta usando `nexo_integral_not_confirmed_for_north_unit`. A fixture R02 não é alterada e R03 não
foi criado.

## Grounding diagnostic D04 — infraestrutura sanitizada, sem execução real

O contrato operacional D04 está documentado em
`docs/evaluation/grounding-diagnostic-d04.md` e a fronteira arquitetural na ADR-011. A execução
continua proibida nesta fase. Os gates offline validam schemas e fixtures sintéticas sem chave,
OpenAI ou Qdrant real:

```powershell
uv run --locked python scripts/validate_grounding_diagnostic_d04.py
uv run --locked python scripts/validate_grounding_diagnostic_d04_result.py
```

Uma futura execução exige flag explícita, case ID P02/P04, usage report privacy-safe e destino D04
novo sob `data/run-reports`. As normalizações são exclusivamente diagnósticas; o status permanece
`grounding_failed`, o exit code permanece 1, não há retry, segunda busca ou segunda geração, a
decisão de qualidade é `NOT_APPLICABLE` e R03 permanece ausente.
