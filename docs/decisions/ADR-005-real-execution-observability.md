# ADR-005 — Observabilidade da execução real controlada

## Contexto

A primeira execução real da Fase 6A precisa materializar o índice local e testar recuperação e
respostas OpenAI sem transformar conteúdo, credenciais ou respostas em logs. A implementação
anterior misturava o limite público em caracteres com `max_output_tokens` e descartava parte do
uso informado pelos provedores. Retries automáticos do SDK também impediam afirmar quantas
tentativas físicas haviam ocorrido.

## Decisão

A futura execução usa `text-embedding-3-small`, 1536 dimensões e `encoding_format="float"` para
embeddings. Respostas usam `gpt-5.6-luna`, Responses API e Structured Outputs com JSON Schema
estrito. Estes identificadores são os modelos definidos para a fase; esta preparação não realiza
chamadas para verificar disponibilidade na conta.

`RAG_MAX_ANSWER_CHARACTERS` continua limitando o contrato público e a validação pós-geração.
`OPENAI_ANSWER_MAX_OUTPUT_TOKENS`, com padrão 1200 e teto defensivo 16384, controla exclusivamente
`max_output_tokens`. A variável não é exposta como parâmetro da CLI nem ao usuário final.

Os providers produzem modelos imutáveis próprios. A telemetria de embeddings contém contagens,
tokens opcionais, lotes e request IDs opacos validados; nunca contém textos ou vetores. A
telemetria de geração contém tokens opcionais, recusa, request ID seguro, chamada lógica,
tentativa física observável e número da tentativa da aplicação; nunca contém prompt, contexto,
resposta bruta ou objetos do SDK.

Tentativas da aplicação e retries de transporte permanecem conceitos separados. Para o primeiro
teste controlado, `OPENAI_MAX_RETRIES=0` e `OPENAI_ANSWER_MAX_RETRIES=0`, tornando uma tentativa
física por chamada lógica auditável. Se retries automáticos forem habilitados, o número físico é
registrado como indisponível até que exista suporte explícito e confiável do SDK.

As CLIs aceitam `--usage-report` e, separadamente, `--overwrite-usage-report`. Relatórios são JSON
UTF-8 publicados atomicamente apenas abaixo de `data/run-reports`, diretório já ignorado por Git.
O caminho rejeita traversal, extensão diferente de JSON, symlink e reparse point quando
detectável. O schema é fechado e uma validação recursiva rejeita campos e valores sensíveis.

O `index-manifest.json` permanece determinístico e separado da observabilidade. Tokens, request
IDs, timestamps e duração existem somente no relatório operacional local e nunca entram no
manifesto versionável.

## Consequências

- As APIs públicas anteriores permanecem compatíveis; variantes `*_with_usage` atendem relatórios.
- Fallback anterior à recuperação registra zero embeddings e zero geração.
- Fallback após recuperação preserva somente a telemetria da busca e zero geração.
- Ausência de usage na API é representada por `null`, sem estimativas inventadas.
- Request IDs fora do formato opaco seguro são descartados.
- Não há preços no código; custos são calculados posteriormente com valores oficiais explícitos.
- O threshold semântico real continua não calibrado e deve ser congelado antes dos smoke tests.

## Limitações

A telemetria não observa tentativas físicas internas quando retries do SDK são maiores que zero.
Validação estrutural e quotes exatas não garantem grounding semântico perfeito. O relatório local
é evidência operacional, não substitui a revisão humana nem o painel oficial de uso. Esta decisão
não executa OpenAI, não cria Qdrant persistente e não gera o manifesto real.
