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

## Autor

João Paulo Silva Borelli
