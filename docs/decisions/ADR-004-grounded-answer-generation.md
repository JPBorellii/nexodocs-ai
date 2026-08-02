# ADR-004 — Geração de respostas fundamentadas

## Decisão

A Fase 5 adiciona geração RAG com prompts versionados (`rag-v1`), Structured Outputs na Responses API e validação local de schema, marcadores e quotes exatas. O cliente OpenAI é injetável; o provedor fake, restrito a `APP_ENV=test`, é determinístico e offline.

O contexto é limitado por caracteres renderizados e as evidências são sempre dados não confiáveis. Preflight bloqueia pedidos clínicos, tentativa de prompt injection, chaves e comandos. Ausência, ambiguidade ou insuficiência de evidência produzem fallback determinístico, sem chamada ao provedor.

As fontes públicas são derivadas exclusivamente do contexto recuperado e são renumeradas por primeira ocorrência. A validação lexical/estrutural garante rastreabilidade e controles determinísticos; não comprova grounding semântico perfeito nem substitui avaliação humana. Há no máximo uma tentativa corretiva planejada para erros estruturais.

## Limitações

Esta decisão não executa chamada OpenAI real, não persiste Qdrant e não inclui interface. A avaliação fake testa contratos, segurança, fallback e determinismo, não fluência ou qualidade semântica de um modelo real. Uma execução real futura requer índice existente, modelo explícito e chave apenas no ambiente.
