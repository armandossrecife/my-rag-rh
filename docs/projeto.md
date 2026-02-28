# 📋 Relatório Técnico: Agente de RH com RAG + Reranking

## Visão Geral do Sistema

Este documento descreve a arquitetura técnica de um **Agente de RH baseado em RAG (Retrieval-Augmented Generation)** desenvolvido inteiramente com bibliotecas nativas Python, sem dependência do LangChain. O sistema permite consultas em documentos PDF de políticas internas de uma organização através de uma interface de terminal.

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FLUXO DO SISTEMA                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [PDFs] → [Leitura] → [Chunking] → [Embeddings] → [ChromaDB]            │
│                                                      │                  │
│  [Pergunta] → [Embedding] → [Similarity Search] ─────┤                  │
│                                                      ↓                  │
│  [Documentos Recuperados] → [Reranking] → [Contexto] → [LLM] → [Resposta]
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📦 1. Importações e Dependências

### Bibliotecas Utilizadas

| Biblioteca | Finalidade | Versão Mínima |
|------------|------------|---------------|
| `pypdf` | Leitura e extração de texto de PDFs | 3.0+ |
| `chromadb` | Banco de dados vetorial para armazenamento e busca | 0.4+ |
| `openai` | Cliente oficial da API OpenAI (embeddings + LLM) | 1.0+ |
| `rich` | Interface de terminal formatada e colorida | 13.0+ |
| `python-dotenv` | Carregamento de variáveis de ambiente | 1.0+ |

### Configuração de Segurança

```python
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit(1)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

**Justificativa:** A chave da API é carregada de um arquivo `.env` para não ser hardcoded no código, seguindo práticas de segurança.

---

## 📂 2. Leitura de Documentos (`carregar_documentos`)

### Funcionamento

```python
def carregar_documentos():
    caminhos = [
        "documentos/politica_ferias.pdf",
        "documentos/politica_home_office.pdf",
        "documentos/codigo_conduta.pdf"
    ]
    # ... processamento
```

### Processo

| Etapa | Descrição |
|-------|-----------|
| 1 | Itera sobre lista pré-definida de caminhos de PDFs |
| 2 | Verifica existência de cada arquivo com `os.path.exists()` |
| 3 | Usa `PdfReader` para abrir e extrair texto página por página |
| 4 | Valida se o texto extraído não está vazio |
| 5 | Armazena em estrutura de dicionário com conteúdo e metadados |

### Estrutura de Dados

```python
{
    "page_content": "Texto extraído da página...",
    "metadata": {
        "documento": "documentos/politica_ferias.pdf",
        "pagina": 1
    }
}
```

### Tratamento de Erros

- Arquivos inexistentes geram aviso mas não interrompem execução
- Páginas com texto vazio são descartadas
- Contagem total é exibida para validação

---

## ✂️ 3. Chunking (`gerar_chunks`)

### Objetivo

Dividir documentos grandes em fragmentos menores para:
- Melhor precisão na recuperação
- Otimização de tokens na API
- Contexto mais relevante para o LLM

### Parâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `chunk_size` | 800 caracteres | Equilíbrio entre contexto e precisão |
| `chunk_overlap` | 150 caracteres | Preserva contexto entre chunks adjacentes |

### Algoritmo

```
1. Split inicial por parágrafos (\n\n)
2. Agrupa parágrafos até atingir chunk_size
3. Chunks excedentes são divididos por caracteres com overlap
4. Metadados são preservados em cada chunk
```

### Exemplo Visual

```
Documento Original (3000 caracteres)
│
├── Chunk 1 (0-800) + Metadados
├── Chunk 2 (650-1450) + Metadados  ← 150 chars overlap
├── Chunk 3 (1300-2100) + Metadados
└── Chunk 4 (1950-2750) + Metadados
```

---

## 🏷️ 4. Enriquecimento de Metadados (`enriquecer_chunks`)

### Classificação Semântica

Cada chunk recebe uma categoria baseada em palavras-chave:

| Categoria | Palavras-chave |
|-----------|----------------|
| `ferias` | "férias", "ferias" |
| `home_office` | "home office", "remoto", "teletrabalho" |
| `conduta` | "conduta", "ética", "etica" |
| `geral` | Default (nenhuma correspondência) |

### Vantagens

- **Filtragem futura:** Possibilidade de filtrar por categoria em queries
- **Debug:** Identificação rápida do tipo de conteúdo nas fontes
- **Transparência:** Usuário sabe de onde vem a informação

---

## 🔢 5. Embeddings (`gerar_embeddings`)

### Tecnologia

- **Modelo:** `text-embedding-3-small` (OpenAI)
- **Dimensões:** 1536 dimensões vetoriais
- **Custo:** Baixo comparado a modelos maiores

### Processo em Batch

```python
def gerar_embeddings(textos: List[str]) -> List[List[float]]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos  # Até 2048 textos por chamada
    )
```

### Otimizações

| Técnica | Benefício |
|---------|-----------|
| Filtragem de textos vazios | Evita chamadas desnecessárias à API |
| Batch de 50 chunks | Reduz número de chamadas HTTP |
| Validação de resposta | Previne erros de parsing |

### Embedding Único (Query)

```python
def gerar_embedding_unico(texto: str) -> List[float]:
    # Para perguntas do usuário
```

---

## 🗄️ 6. Vector Store (`inicializar_vectorstore`)

### ChromaDB Nativo

| Componente | Configuração |
|------------|--------------|
| Cliente | `PersistentClient` |
| Path | `./chroma_rh` |
| Coleção | `rh_documentos` |
| Similaridade | Cosseno (`hnsw:space: cosine`) |

### Fluxo de Inicialização

```
1. Verifica existência do diretório
2. Limpa banco existente (garante integridade)
3. Carrega documentos → Gera chunks → Enriquece metadados
4. Gera IDs únicos (hash MD5 do conteúdo)
5. Cria embeddings em batch
6. Insere no ChromaDB com metadatas
7. Valida count final
```

### Estrutura de Inserção

```python
collection.add(
    ids=batch_ids,           # IDs únicos (hash)
    embeddings=embeddings,   # Vetores 1536D
    documents=batch_textos,  # Texto original
    metadatas=batch_metadatas # Metadados estruturados
)
```

### IDs Únicos

```python
chunk_id = f"chunk_{hashlib.md5(chunk['page_content'].encode()).hexdigest()[:16]}"
```

**Justificativa:** Garante que chunks idênticos não sejam duplicados e permite upsert seguro.

---

## 🔍 7. Recuperação (`responder_pergunta` - Parte 1)

### Query Vetorial

```python
resultados = collection.query(
    query_embeddings=[pergunta_embedding],
    n_results=8,
    include=["documents", "metadatas", "distances"]
)
```

### Parâmetros

| Parâmetro | Valor | Função |
|-----------|-------|--------|
| `query_embeddings` | `[embedding]` | Vetor da pergunta |
| `n_results` | 8 | Recupera candidatos para reranking |
| `include` | documents, metadatas, distances | Dados retornados |

### Estrutura de Resposta

```python
{
    "documents": [["chunk1", "chunk2", ...]],
    "metadatas": [[{...}, {...}, ...]],
    "distances": [[0.15, 0.23, ...]]
}
```

### Validação

- Verifica se `documents[0]` não está vazio
- Filtra chunks com texto vazio
- Preserva metadados associados

---

## 📊 8. Reranking (`rerank_documentos`)

### Conceito

O **reranking** é uma etapa crítica que melhora a precisão da recuperação:

| Etapa | Método | Precisão |
|-------|--------|----------|
| Recuperação inicial | Similaridade vetorial | ~70-80% |
| Reranking | LLM semântico | ~90-95% |

### Prompt de Avaliação

```python
prompt = f"""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{doc["page_content"][:500]}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
```

### Processo

```
1. Para cada documento recuperado (até 8)
2. Chama LLM com prompt de avaliação
3. Parse da resposta numérica (0-10)
4. Ordena por score decrescente
5. Retorna lista reordenada
```

### Otimizações

| Técnica | Benefício |
|---------|-----------|
| `max_tokens=5` | Limita resposta ao número, reduz custo |
| `temperature=0` | Respostas determinísticas |
| Truncamento 500 chars | Reduz tokens no prompt |
| Try/except | Previne falha total se LLM errar |

### Barra de Progresso

```python
with Progress(...) as progress:
    task = progress.add_task("Realizando Reranking...", total=len(documentos))
    # Update a cada documento processado
```

---

## 🤖 9. Geração de Resposta (`responder_pergunta` - Parte 2)

### Construção do Contexto

```python
contexto_final = documentos_rerankeados[:4]  # Top 4 após reranking
contexto_texto = "\n\n".join([doc["page_content"] for doc in contexto_final])
```

### Prompt Final

```python
prompt_final = f"""
Você é um agente de RH corporativo.
Responda APENAS com base nas políticas internas abaixo.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}
"""
```

### Configuração do LLM

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| `model` | `gpt-4o-mini` | Custo-benefício ótimo |
| `temperature` | 0 | Respostas consistentes |
| `messages` | `[{"role": "user", ...}]` | Formato Chat Completions |

### Retorno

```python
return resposta, contexto_final
# resposta: str (texto da resposta)
# contexto_final: List[Dict] (fontes usadas)
```

---

## 🎨 10. Interface de Terminal (Rich)

### Componentes Visuais

| Componente | Uso | Exemplo |
|------------|-----|---------|
| `Console` | Saída formatada | `console.print()` |
| `Panel` | Agrupamento visual | Boxes com bordas |
| `Markdown` | Renderização de resposta | Formatação LLM |
| `Progress` | Barras de progresso | Reranking |
| `Status` | Spinner de loading | "Consultando..." |
| `Syntax` | Highlight de código | Trechos de documentos |

### Exemplo de Saída

```
╭────────────────────────────────────────────────────────────────────╮
│  🤖 AGENTE DE RH — POLÍTICAS INTERNAS                              │
│  RAG + Reranking com ChromaDB Nativo                               │
╰────────────────────────────────────────────────────────────────────╯

👤 Você: Quais os passos para eu tirar férias?

⏳ Consultando políticas internas...

╭─────────────────────────── 🤖 Agente ────────────────────────────╮
│                                                                  │
│  Para solicitar férias, siga os passos abaixo:                   │
│  1. Verifique seu saldo no portal...                             │
│                                                                  │
╰──────────────────────────────────────────────────────────────────╯

╭───────────────────── 📚 FONTES UTILIZADAS ──────────────────────╮
│  Trecho 1                                                       │
│  Documento: documentos/politica_ferias.pdf                      │
│  Categoria: ferias                                              │
│  [Conteúdo do trecho...]                                        │
╰─────────────────────────────────────────────────────────────────╯
```

---

## 🔄 11. Loop Principal (`main`)

### Fluxo de Execução

```python
def main():
    1. limpar_tela()
    2. imprimir_cabecalho()
    3. inicializar_vectorstore()
    4. while True:
        a. console.input() ← Pergunta do usuário
        b. if sair: break
        c. responder_pergunta()
        d. console.print(Panel(Markdown(resposta)))
        e. imprimir_fontes(fontes)
```

### Tratamento de Erros

| Cenário | Ação |
|---------|------|
| `KeyboardInterrupt` | Mensagem amigável de saída |
| Erro na query | Log do erro + continua loop |
| Erro crítico | Stack trace + exit(1) |
| Input vazio | Ignora e pede nova pergunta |

---

## 📊 12. Métricas e Performance

### Tempos Estimados

| Operação | Tempo Médio |
|----------|-------------|
| Carregar PDFs (3 docs) | 2-5 segundos |
| Gerar chunks | < 1 segundo |
| Embeddings (100 chunks) | 10-20 segundos |
| Query + Reranking (8 docs) | 15-30 segundos |
| Geração de resposta | 2-5 segundos |

### Custos Estimados (OpenAI API)

| Operação | Tokens | Custo Aprox. |
|----------|--------|--------------|
| Embeddings (100 chunks × 800 chars) | ~20.000 | $0.002 |
| Reranking (8 docs × 500 chars) | ~4.000 | $0.0004 |
| Resposta final | ~1.500 | $0.0002 |
| **Total por query** | - | **~$0.003** |

---

## 🔐 13. Considerações de Segurança

| Aspecto | Implementação |
|---------|---------------|
| API Key | Arquivo `.env` (não commitado) |
| Dados sensíveis | Apenas políticas públicas de RH |
| Persistência | Local (`./chroma_rh`) |
| Logs | Sem dados de usuários armazenados |

---

## 🚀 14. Possíveis Melhorias Futuras

| Melhoria | Impacto | Complexidade |
|----------|---------|--------------|
| Cache de embeddings | Reduz custo API | Baixa |
| Reranking com modelo dedicado | Mais preciso | Média |
| Multi-tenant | Suporte a múltiplas orgs | Alta |
| API REST | Integração com outros sistemas | Média |
| Dashboard web | Interface gráfica | Média |
| Hybrid search (texto + vetorial) | Melhor recall | Média |

---

## 📁 15. Estrutura de Arquivos Recomendada

```
rag-rh/
├── main_cli3.py          # Script principal
├── .env                  # Variáveis de ambiente (gitignore)
├── pyproject.toml        # Dependências do projeto
├── documentos/
│   ├── politica_ferias.pdf
│   ├── politica_home_office.pdf
│   └── codigo_conduta.pdf
└── chroma_rh/            # Banco vetorial (gitignore)
    └── rh_documentos/
```

---

## ✅ Conclusão

Este sistema demonstra uma implementação **protótipo** de RAG sem dependências pesadas como LangChain. As principais vantagens são:

1. **Controle total** sobre cada etapa do pipeline
2. **Menos dependências** = menos vulnerabilidades
3. **Custo otimizado** com batching e modelos eficientes
4. **Debug facilitado** com logs detalhados
5. **UX profissional** com interface Rich no terminal

A arquitetura é escalável e pode ser adaptada para outros domínios além de RH, bastando alterar os documentos de entrada e prompts.
