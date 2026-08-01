# ADR-001: geração da base documental fictícia

## Contexto e decisão

Os documentos da demonstração são definidos em JSON canônico e validados por JSON Schema
Draft 2020-12. Um pacote Python modular gera três PDFs e dois CSVs, além de um catálogo com
hashes. ReportLab usa `invariant=1`, fontes padrão Helvetica e metadados fixos; não há fontes
externas. Os CSVs usam UTF-8 com BOM e ponto e vírgula.

## Consequências

Os modos `--write` e `--check` tornam a geração determinística e verificável offline. O catálogo
não possui timestamp, hostname ou dados do ambiente. A validação confirma schema, conteúdo,
assinaturas, hashes e sincronização. A reprodutibilidade depende das versões preservadas no lock.
FPDF, HTML e fontes externas foram rejeitados. No futuro, os artefatos poderão ser entradas de RAG,
sem que esta decisão implemente ingestão, embeddings ou busca.
