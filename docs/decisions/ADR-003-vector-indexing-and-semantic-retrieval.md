# ADR-003: indexação vetorial e recuperação semântica estruturada

## Contexto

Os chunks determinísticos da base fictícia precisam ser recuperáveis por significado sem perder
rastreabilidade, fontes ou as garantias offline das fases anteriores. A camada desta fase prepara
embeddings e um índice vetorial, mas ainda não gera respostas em linguagem natural. Documentos
continuam sendo dados, e a ausência de evidência deve ser representada explicitamente.

As dependências diretas escolhidas são `openai==2.52.0` e `qdrant-client==1.18.0`, preservadas no
lock do projeto. Não foram adotados frameworks de RAG: a integração usa um `Protocol` pequeno para
provedores de embeddings e o cliente Qdrant diretamente.

## Decisão

O contrato de embeddings expõe nome do provedor, modelo, dimensão, embedding em lote e embedding de
consulta. O provedor OpenAI usa `text-embedding-3-small` por padrão, cliente injetável, lotes
determinísticos e valida quantidade, ordem, dimensão e finitude dos vetores. Sua execução é somente
manual e requer configuração válida no ambiente do processo.

Um provedor lexical falso usa hashing determinístico de tokens e bigramas, sinal por bucket e
normalização L2. Ele existe apenas para testes e avaliação offline, não representa qualidade
semântica de produção e é recusado quando `APP_ENV` não é `test`.

Qdrant opera em três modos:

- `memory`, efêmero e permitido somente em testes, sem rede;
- `local`, persistente no caminho configurado por `QDRANT_PATH`;
- `remote`, reservado a operação manual, com URL e credenciais fornecidas pelo ambiente.

A coleção padrão é `nexodocs_chunks_v1`, com distância Cosine e dimensão igual à do provedor. Uma
coleção existente incompatível falha em vez de ser recriada. Cada ponto usa
`uuid5(NEXODOCS_CHUNK_NAMESPACE, chunk_id)`, portanto sua identidade não depende de ordem,
timestamp ou máquina.

O `index-plan.json` é um artefato versionado, determinístico e gerável offline. Ele registra fontes,
hashes, identidade e compatibilidade esperada, sem vetores, textos integrais, caminhos absolutos ou
segredos. O `index-manifest.json` é reservado para uma futura indexação real concluída com sucesso e
não é produzido pelos gates desta fase.

A indexação real é incremental: um ponto só é reutilizado quando identidade, conteúdo, provedor,
modelo, dimensão, schema e manifesto de processamento continuam compatíveis. Pontos novos ou
incompatíveis recebem embedding e upsert; pontos ausentes da fonte são apenas marcados como
obsoletos. Remoção exige o subcomando `prune` e a confirmação do nome exato da coleção, nunca apaga a
coleção inteira e não ocorre no CI.

A recuperação aceita `document_id`, `category`, `source_format`, `owner_area`, `version` e
`classification`. Ela limita `top_k`, permite threshold opcional, deduplica chunks, limita a
concentração por documento e preserva texto, score, metadados, localizador e rótulo de citação. Sem
resultado acima dos critérios, retorna `status="no_evidence"` e
`reason="no_match_or_below_threshold"`; não tenta completar a resposta por conhecimento próprio.

## Validação e segurança

O plano é conferido byte a byte no CI. A avaliação de recuperação usa somente o embedder lexical,
Qdrant em memória e casos fictícios, medindo Hit Rate@k, Recall@k, MRR e precisão de fallback com
limites iniciais modestos. Nenhuma chave é aceita pela CLI, `.env` não é carregado automaticamente e
as chaves permanecem vazias em `.env.example`. Testes e quality gates não fazem chamadas à OpenAI,
não conectam a Qdrant remoto e não registram vetores, textos integrais ou configuração sensível.

## Consequências e limitações

O plano reproduzível pode ser revisado sem custo ou infraestrutura, enquanto escrita, inspeção,
prune e busca no índice real permanecem operações deliberadamente manuais. Trocas de modelo,
dimensão ou distância exigem nova coleção ou migração aprovada; não há migração nem remoção
automática. A avaliação lexical não estima a qualidade dos embeddings OpenAI, e esta fase não inclui
OCR, reranking, geração por LLM, interface web nem validação de infraestrutura remota. Uma fase futura
poderá gerar respostas estritamente fundamentadas nos resultados e citações deste contrato.
