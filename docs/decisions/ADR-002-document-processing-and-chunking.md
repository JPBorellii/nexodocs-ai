# ADR-002: processamento determinístico e chunking documental

## Contexto e decisão

A base fictícia versionada torna-se entrada rastreável para uma futura recuperação semântica. O
pipeline valida catálogo e fontes, extrai PDFs com `pypdf` página a página e CSVs registro a
registro. O texto recebe normalização conservadora; chunks PDF usam seções e parágrafos, alvo de
1.000, máximo de 1.200 e sobreposição de até 150 caracteres somente na mesma seção. Não há
cruzamento de páginas, seções ou linhas CSV.

Cada linha UTF-8 de `chunks.jsonl` tem ID determinístico, hash e localizador. `manifest.json`
registra hashes, estratégia e totais. `--write` e `--check` usam staging temporário para escrita
atômica e comparação byte a byte offline.

## Consequências

Não há OCR, IA, embeddings ou banco vetorial. PDFs sem texto extraível falham explicitamente.
`pypdf` e `jsonschema` são dependências de execução; ReportLab permanece para geração em
desenvolvimento. Uma etapa futura poderá gerar embeddings sem alterar fontes nem este contrato.
