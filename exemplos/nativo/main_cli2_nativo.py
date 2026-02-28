# ============================================
# AGENTE DE RH COM RAG + RERANKING (VERSÃO CLI NATIVA - DEBUG)
# ChromaDB + OpenAI API + Terminal + Rich
# ============================================

# =========================
# 1. IMPORTAÇÕES
# =========================

import os
import sys
import hashlib
import shutil
from typing import List, Dict
from dotenv import load_dotenv

from pypdf import PdfReader
import chromadb
from openai import OpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console()

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    console.print("[bold red]ERRO:[/bold red] OPENAI_API_KEY não encontrada no arquivo .env")
    sys.exit(1)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# 2. CONFIGURAÇÕES GERAIS
# =========================

PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# =========================
# 3. LEITURA DOS DOCUMENTOS
# =========================

def carregar_documentos(lista_documentos: List[str] = None) -> List[Dict]:
    
    documentos = []

    with console.status("[bold green]Carregando documentos PDF..."):
        for caminho in lista_documentos:
            if not os.path.exists(caminho):
                console.print(f"[yellow]AVISO:[/yellow] Arquivo não encontrado: {caminho}")
                continue
            
            reader = PdfReader(caminho)
            
            for i, page in enumerate(reader.pages):
                texto = page.extract_text()
                if texto and texto.strip():
                    documentos.append({
                        "page_content": texto.strip(),
                        "metadata": {
                            "documento": caminho,
                            "pagina": i + 1
                        }
                    })
    
    console.print(f"[green]✓[/green] OK ([bold]{len(documentos)}[/bold] páginas carregadas)")
    return documentos

# =========================
# 4. CHUNKING
# =========================

def gerar_chunks(documentos: List[Dict], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Dict]:
    chunks = []
    
    for doc in documentos:
        texto = doc["page_content"]
        metadata = doc["metadata"]
        
        # Split por parágrafos
        paragrafos = texto.split('\n\n')
        chunk_atual = ""
        
        for paragrafo in paragrafos:
            paragrafo = paragrafo.strip()
            if not paragrafo:
                continue
                
            if len(chunk_atual) + len(paragrafo) <= chunk_size:
                chunk_atual += paragrafo + " "
            else:
                if chunk_atual.strip():
                    chunks.append({
                        "page_content": chunk_atual.strip(),
                        "metadata": metadata.copy()
                    })
                chunk_atual = paragrafo + " "
        
        if chunk_atual.strip():
            chunks.append({
                "page_content": chunk_atual.strip(),
                "metadata": metadata.copy()
            })
    
    # Divide chunks muito grandes
    chunks_finais = []
    for chunk in chunks:
        if len(chunk["page_content"]) > chunk_size:
            texto = chunk["page_content"]
            for i in range(0, len(texto), chunk_size - chunk_overlap):
                chunk_texto = texto[i:i + chunk_size]
                if chunk_texto.strip():
                    chunks_finais.append({
                        "page_content": chunk_texto.strip(),
                        "metadata": chunk["metadata"].copy()
                    })
        else:
            if chunk["page_content"].strip():
                chunks_finais.append(chunk)
    
    return chunks_finais

# =========================
# 5. ENRIQUECIMENTO COM METADADOS
# =========================

def enriquecer_chunks(chunks: List[Dict]) -> List[Dict]:
    for chunk in chunks:
        texto = chunk["page_content"].lower()

        if "férias" in texto or "ferias" in texto:
            chunk["metadata"]["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto or "teletrabalho" in texto:
            chunk["metadata"]["categoria"] = "home_office"
        elif "conduta" in texto or "ética" in texto or "etica" in texto:
            chunk["metadata"]["categoria"] = "conduta"
        else:
            chunk["metadata"]["categoria"] = "geral"

    return chunks

# =========================
# 6. EMBEDDINGS
# =========================

def gerar_embeddings(textos: List[str]) -> List[List[float]]:
    # Filtra textos vazios
    textos_validos = [t for t in textos if t and t.strip()]
    if not textos_validos:
        return []
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos_validos
    )
    return [embedding.embedding for embedding in response.data]

def gerar_embedding_unico(texto: str) -> List[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto
    )
    return response.data[0].embedding

# =========================
# 7. VECTOR STORE
# =========================

def inicializar_vectorstore(lista_documentos: List[str]) -> chromadb.api.models.Collection.Collection:
    # Força recriação para garantir dados limpos
    if os.path.exists(PERSIST_DIRECTORY):
        console.print("[yellow]![/yellow] Banco existente detectado. Limpando para garantir integridade...")
        shutil.rmtree(PERSIST_DIRECTORY)
    
    chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    console.print("[yellow]![/yellow] Criando novo banco vetorial...")
    documentos = carregar_documentos(lista_documentos)
    
    if not documentos:
        console.print("[bold red]ERRO:[/bold red] Nenhum documento carregado. Verifique a pasta 'documentos'.")
        console.print(f"[dim]Caminho esperado: {os.path.abspath('documentos')}[/dim]")
        sys.exit(1)
    
    chunks = gerar_chunks(documentos)
    console.print(f"[green]✓[/green] [bold]{len(chunks)}[/bold] chunks gerados")
    
    chunks = enriquecer_chunks(chunks)
    
    with console.status("[bold green]Criando embeddings e salvando banco..."):
        # Deleta coleção se existir
        try:
            chroma_client.delete_collection(name="rh_documentos")
        except:
            pass
        
        collection = chroma_client.create_collection(
            name="rh_documentos",
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = []
        documentos_textos = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            if not chunk["page_content"].strip():
                continue
            chunk_id = f"chunk_{hashlib.md5(chunk['page_content'].encode()).hexdigest()[:16]}"
            ids.append(chunk_id)
            documentos_textos.append(chunk["page_content"])
            metadatas.append(chunk["metadata"])
        
        console.print(f"[dim]Total de documentos para inserção: {len(ids)}[/dim]")
        
        batch_size = 50
        total_inserido = 0
        
        for i in range(0, len(documentos_textos), batch_size):
            batch_textos = documentos_textos[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            embeddings = gerar_embeddings(batch_textos)
            
            if embeddings:
                collection.add(
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_textos,
                    metadatas=batch_metadatas
                )
                total_inserido += len(batch_ids)
        
        console.print(f"[green]✓[/green] OK ([bold]{total_inserido}[/bold] chunks inseridos)")
        
        # Verifica quantos documentos existem na coleção
        count = collection.count()
        console.print(f"[dim]📊 Total na coleção: {count} documentos[/dim]")

    return collection

# =========================
# 8. RERANKING
# =========================

def rerank_documentos(pergunta: str, documentos: List[Dict], client: OpenAI) -> List[Dict]:
    if not documentos:
        console.print("[yellow]⚠️[/yellow] Nenhum documento para reranking")
        return []
    
    documentos_com_score = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Realizando Reranking...", total=len(documentos))
        
        for doc in documentos:
            prompt = f"""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{doc["page_content"][:500]}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
            try:
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=5
                )
                
                score_text = response.choices[0].message.content.strip()
                
                try:
                    score = float(score_text)
                except:
                    score = 0
            except Exception as e:
                console.print(f"[yellow]⚠️[/yellow] Erro no reranking: {e}")
                score = 0

            documentos_com_score.append((score, doc))
            progress.update(task, advance=1)

    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    console.print("[green]✓[/green] Reranking concluído")
    return [doc for _, doc in documentos_ordenados]

# =========================
# 9. PIPELINE RAG
# =========================

def responder_pergunta(pergunta: str, collection) -> tuple[str, List[Dict]]:
    console.print(f"[dim]🔍 Buscando por: '{pergunta[:50]}...'[/dim]")
    
    # Gera embedding da pergunta
    pergunta_embedding = gerar_embedding_unico(pergunta)
    console.print(f"[dim]📐 Embedding gerado: {len(pergunta_embedding)} dimensões[/dim]")

    # Recuperação
    resultados = collection.query(
        query_embeddings=[pergunta_embedding],
        n_results=8,
        include=["documents", "metadatas", "distances"]
    )

    console.print(f"[dim]📦 Resultados da query: {resultados}[/dim]")
    
    if not resultados.get("documents") or not resultados["documents"][0]:
        console.print("[yellow]⚠️[/yellow] Nenhum documento recuperado do banco vetorial")
        return "Não encontrei informações relevantes nos documentos.", []

    documentos_recuperados = []
    for i, doc_text in enumerate(resultados["documents"][0]):
        if doc_text and doc_text.strip():
            documentos_recuperados.append({
                "page_content": doc_text,
                "metadata": resultados["metadatas"][0][i] if resultados.get("metadatas") and resultados["metadatas"][0] else {}
            })

    console.print(f"[dim]📄 Documentos recuperados: {len(documentos_recuperados)}[/dim]")
    
    if not documentos_recuperados:
        return "Não encontrei informações relevantes nos documentos.", []

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        client
    )

    contexto_final = documentos_rerankeados[:4]
    
    console.print(f"[dim]🎯 Contexto final: {len(contexto_final)} documentos[/dim]")

    contexto_texto = "\n\n".join(
        [doc["page_content"] for doc in contexto_final]
    )

    prompt_final = f"""
Você é um agente de RH corporativo.
Responda APENAS com base nas políticas internas abaixo.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}
"""

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt_final}],
        temperature=0
    )

    resposta = response.choices[0].message.content

    return resposta, contexto_final

# =========================
# 10. INTERFACE
# =========================

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimir_cabecalho():
    console.print(Panel.fit(
        "[bold blue]🤖 AGENTE DE RH — POLÍTICAS INTERNAS[/bold blue]\n"
        "[dim]RAG + Reranking com ChromaDB Nativo[/dim]",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print("\nDigite sua pergunta ou '[bold]sair[/bold]' para encerrar.\n")

def imprimir_fontes(fontes: List[Dict]):
    console.print(Panel(
        "[bold]📚 FONTES UTILIZADAS[/bold]",
        border_style="yellow",
        padding=(0, 1)
    ))
    
    for i, doc in enumerate(fontes, start=1):
        console.print(f"\n[bold cyan]Trecho {i}[/bold cyan]")
        console.print(f"  [dim]Documento:[/dim] {doc['metadata'].get('documento', 'Desconhecido')}")
        console.print(f"  [dim]Categoria:[/dim] {doc['metadata'].get('categoria', 'Geral')}")
        
        syntax = Syntax(
            doc["page_content"][:200] + ("..." if len(doc["page_content"]) > 200 else ""),
            "text",
            theme="monokai",
            line_numbers=False,
            word_wrap=True
        )
        console.print(syntax)

def main():
    limpar_tela()
    imprimir_cabecalho()
    
    caminhos_documentos = [
        "documentos/politica_ferias.pdf",
        "documentos/politica_home_office.pdf",
        "documentos/codigo_conduta.pdf"
    ]
    
    try:
        collection = inicializar_vectorstore(caminhos_documentos)
    except Exception as e:
        console.print(Panel(f"[bold red]ERRO CRÍTICO:[/bold red] {e}", border_style="red"))
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)

    console.print("\n[bold green]✅ Sistema pronto para consultas.[/bold green]\n")

    while True:
        try:
            pergunta = console.input("[bold green]👤 Você:[/bold green] ").strip()

            if pergunta.lower() in ["sair", "exit", "quit"]:
                console.print("\n[bold blue]👋 Encerrando agente de RH. Até logo![/bold blue]")
                break
            
            if not pergunta:
                continue

            with console.status("[bold green]Consultando políticas internas...", spinner="dots"):
                try:
                    resposta, fontes = responder_pergunta(pergunta, collection)
                except Exception as e:
                    console.print(f"\n[bold red]❌ Erro ao processar a pergunta:[/bold red] {e}")
                    import traceback
                    console.print(f"[dim]{traceback.format_exc()}[/dim]")
                    continue

            console.print()
            console.print(Panel(
                Markdown(resposta, code_theme="monokai"),
                title="[bold blue]🤖 Agente[/bold blue]",
                border_style="blue",
                padding=(1, 2)
            ))
            
            if fontes:
                imprimir_fontes(fontes)
            else:
                console.print("\n[yellow]⚠️  Nenhuma fonte específica foi utilizada para esta resposta.[/yellow]")
                    
            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]👋 Interrupção detectada. Encerrando...[/bold yellow]")
            break

if __name__ == "__main__":
    main()