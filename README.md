# NexoDocs AI

> Encontre. Confirme. Decida.

Agente corporativo de inteligência artificial da empresa fictícia **Nexo Saúde Integrada**.

O projeto utiliza Retrieval-Augmented Generation (RAG) para responder perguntas de colaboradores com base em documentos corporativos, exibindo as fontes consultadas e informando claramente quando uma resposta não estiver disponível na base de conhecimento.

## Status

🚧 Projeto em desenvolvimento para o Challenge Alura Agente — ONE AI for Tech.

## Objetivo

Construir uma solução que:

- processe documentos corporativos;
- gere embeddings;
- realize recuperação semântica;
- responda somente com base nas fontes recuperadas;
- cite os documentos utilizados;
- evite respostas inventadas;
- seja executada em nuvem na Oracle Cloud Infrastructure.

## Tecnologias planejadas

- Python
- OpenAI API
- LangChain
- LangGraph
- Qdrant
- Streamlit
- Docker
- Oracle Cloud Infrastructure
- GitHub Actions

## Aviso

Todos os nomes, documentos, pessoas, políticas e dados utilizados neste projeto serão fictícios e destinados exclusivamente a fins educacionais.

## Desenvolvimento

Pré-requisitos: `uv 0.11.32` e Python 3.14. No PowerShell, prepare o ambiente com
`./scripts/bootstrap.ps1` e execute todos os quality gates com `./scripts/quality.ps1`.
O CI executa os mesmos gates em cada pull request e nos pushes para `main`.

Para corrigir problemas locais do Ruff, use `uv run --locked ruff format .` e
`uv run --locked ruff check --fix .`.

## Base documental fictícia

A base inicial está em `knowledge_base/`: os cinco documentos são definidos em JSON
canônico, validados por schema e gerados deterministicamente como três PDFs e dois CSVs.
Os PDFs cobrem cancelamentos e reagendamentos, férias e benefícios, e privacidade; os
CSVs cobrem convênios por unidade e o diretório de áreas. Todo conteúdo é fictício e
educacional.

Gere a base com `./scripts/generate_knowledge_base.ps1`, valide-a com
`uv run --locked python scripts/validate_knowledge_base.py` e confira sua sincronização
com `uv run --locked python scripts/generate_knowledge_base.py --check`. O catálogo em
`knowledge_base/metadata/catalog.json` registra metadados, hashes, tamanhos e localizadores.
O CI executa a sincronização e a validação nos quality gates. A reprodução de bytes depende
das versões bloqueadas das bibliotecas; futuras atualizações exigem revisão explícita.

## Processamento determinístico

A Fase 3 transforma somente os PDFs e CSVs versionados em `knowledge_base/processed/`.
`chunks.jsonl` contém um chunk rastreável por linha, com fonte, página/seção ou linha CSV,
hashes e metadados documentais. `manifest.json` registra a estratégia, totais e hashes.

Use `./scripts/process_knowledge_base.ps1` para gerar e validar. Para conferir sem escrever,
execute `uv run --locked python scripts/process_knowledge_base.py --check`; para validar,
execute `uv run --locked python scripts/validate_processed_knowledge_base.py`. O pipeline é
offline e não cruza páginas PDF nem linhas CSV. A extração depende da camada textual disponível;
OCR não faz parte desta fase. O CI executa ambos os checks.

## Recuperação vetorial

A camada de recuperação transforma os chunks processados em pontos identificados por UUIDv5, usa
embeddings validados e consulta uma coleção Qdrant com distância Cosine. O `Retriever` devolve JSON
estruturado com texto, score, metadados e citações; ele não gera respostas por LLM. Filtros aceitos:
`document_id`, `category`, `source_format`, `owner_area`, `version` e `classification`. Quando não há
evidência compatível com filtros e threshold opcional, o resultado usa `status: "no_evidence"`.

O plano versionado descreve os pontos e hashes esperados sem conter vetores ou textos integrais. Ele
é diferente do índice real no Qdrant e pode ser gerado, conferido e validado totalmente offline:

```powershell
uv run --locked python scripts/index_knowledge_base.py plan --write
uv run --locked python scripts/index_knowledge_base.py plan --check
uv run --locked python scripts/validate_index_plan.py
```

Escrita incremental, inspeção, remoção de obsoletos e busca são operações manuais sobre o ambiente
explicitamente configurado. `prune` exige o nome exato da coleção e não apaga a coleção inteira:

```powershell
uv run --locked python scripts/index_knowledge_base.py write
uv run --locked python scripts/index_knowledge_base.py check-index
uv run --locked python scripts/index_knowledge_base.py prune --confirm-collection nexodocs_chunks_v1
uv run --locked python scripts/search_knowledge_base.py "Como remarcar uma consulta?" --top-k 5
```

Configure apenas variáveis do processo conforme `.env.example`; a aplicação não carrega `.env`
automaticamente, as chaves de OpenAI e Qdrant ficam vazias no repositório e nenhuma credencial é
aceita por argumento de linha de comando. `EMBEDDING_PROVIDER=openai` oferece embeddings semânticos
para operação real. O provedor lexical `fake` e `QDRANT_MODE=memory` são restritos a `APP_ENV=test`;
`QDRANT_MODE=local` persiste em `QDRANT_PATH`, e `remote` exige `QDRANT_URL` e configuração segura no
ambiente.

O CI executa `plan --check` e uma avaliação offline com fake lexical e Qdrant em memória, sem OpenAI,
rede ou segredos. Essa avaliação não mede qualidade semântica real. A fase também não inclui OCR,
reranking ou geração de respostas, não remove pontos automaticamente e exige nova coleção ou
migração explícita quando modelo, dimensão ou distância se tornam incompatíveis. A decisão completa
está em `docs/decisions/ADR-003-vector-indexing-and-semantic-retrieval.md`.

## Autor

## Fase 5 — respostas fundamentadas

O RAG agora possui contratos JSON, prompts versionados, contexto limitado por caracteres, citações com quotes exatas e fallbacks determinísticos. Evidências e perguntas são dados não confiáveis; pedidos clínicos, de segredo, prompt ou execução de comandos são bloqueados antes da geração.

Valide os contratos com `uv run --locked python scripts/validate_rag_contracts.py` e execute a avaliação offline com `APP_ENV=test uv run --locked python scripts/evaluate_rag.py`. O fake é somente para testes; uma resposta OpenAI real é procedimento manual e exige índice existente, modelo explícito e chave somente no ambiente. A avaliação offline mede contratos, rastreabilidade e segurança, não qualidade semântica real ou avaliação humana.

O CI continua executando `quality.ps1`, que inclui ambas as verificações.

## Preparação da execução real

A futura Fase 6A usa limites separados para caracteres públicos e tokens da Responses API,
telemetria tipada sem conteúdo e relatórios opcionais ignorados em `data/run-reports`. O primeiro
teste controlado desativa retries de transporte para tornar chamadas auditáveis. Consulte a
decisão em `docs/decisions/ADR-005-real-execution-observability.md` e o procedimento completo em
`docs/runbooks/phase-6a-real-local-execution.md`. O threshold semântico real ainda precisa ser
calibrado e congelado antes dos smoke tests.

João Paulo Silva Borelli
