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

## Autor

João Paulo Silva Borelli
