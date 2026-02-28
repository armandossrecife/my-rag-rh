# ============================================
# AGENTE DE RH COM RAG + RERANKING (VERSÃO CLI TURBINADA)
# LangChain + Terminal + Rich
# ============================================

# =========================
# 1. IMPORTAÇÕES
# =========================

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

# Rich imports
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.syntax import Syntax

console = Console()

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    console.print("[bold red]ERRO:[/bold red] OPENAI_API_KEY não encontrada no arquivo .env")
    sys.exit(1)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# =========================
# 2. CONFIGURAÇÕES GERAIS
# =========================

PERSIST_DIRECTORY = "./chroma_rh"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# =========================
# 3. LEITURA DOS DOCUMENTOS
# =========================

def carregar_documentos():
    caminhos = [
        "documentos/politica_ferias.pdf",
        "documentos/politica_home_office.pdf",
        "documentos/codigo_conduta.pdf"
    ]

    documentos = []

    with console.status("[bold green]Carregando documentos PDF..."):
        for caminho in caminhos:
            if not os.path.exists(caminho):
                console.print(f"[yellow]AVISO:[/yellow] Arquivo não encontrado: {caminho}")
                continue
            loader = PyPDFLoader(caminho)
            docs = loader.load()

            for doc in docs:
                doc.metadata["documento"] = caminho

            documentos.extend(docs)
    
    console.print(f"[green]✓[/green] OK ([bold]{len(documentos)}[/bold] páginas carregadas)")
    return documentos

# =========================
# 4. CHUNKING
# =========================

def gerar_chunks(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documentos)

# =========================
# 5. ENRIQUECIMENTO COM METADADOS
# =========================

def enriquecer_chunks(chunks):
    for chunk in chunks:
        texto = chunk.page_content.lower()

        if "férias" in texto:
            chunk.metadata["categoria"] = "ferias"
        elif "home office" in texto or "remoto" in texto:
            chunk.metadata["categoria"] = "home_office"
        elif "conduta" in texto or "ética" in texto:
            chunk.metadata["categoria"] = "conduta"
        else:
            chunk.metadata["categoria"] = "geral"

    return chunks

# =========================
# 6. VECTOR STORE
# =========================

def inicializar_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        with console.status("[bold green]Carregando banco vetorial existente..."):
            vectorstore = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=embeddings
            )
        console.print("[green]✓[/green] OK")
    else:
        console.print("[yellow]![/yellow] Banco vetorial não encontrado. Processando documentos...")
        documentos = carregar_documentos()
        if not documentos:
            console.print("[bold red]ERRO:[/bold red] Nenhum documento carregado. Verifique a pasta 'documentos'.")
            sys.exit(1)
            
        chunks = gerar_chunks(documentos)
        chunks = enriquecer_chunks(chunks)
        
        with console.status("[bold green]Criando embeddings e salvando banco..."):
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
        console.print("[green]✓[/green] OK")

    return vectorstore

# =========================
# 7. RERANKING COM BARRA DE PROGRESSO
# =========================

def rerank_documentos(pergunta, documentos, llm):
    prompt_rerank = PromptTemplate(
        input_variables=["pergunta", "texto"],
        template="""
Você é um especialista em políticas internas de RH.

Pergunta do usuário:
{pergunta}

Trecho do documento:
{texto}

Avalie a relevância desse trecho para responder a pergunta.
Responda apenas com um número de 0 a 10.
"""
    )

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
            score = llm.invoke(
                prompt_rerank.format(
                    pergunta=pergunta,
                    texto=doc.page_content
                )
            ).content

            try:
                score = float(score)
            except:
                score = 0

            documentos_com_score.append((score, doc))
            progress.update(task, advance=1)

    # ✅ CORREÇÃO: Ordenar os documentos pelo score
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    console.print("[green]✓[/green] Reranking concluído")
    return [doc for _, doc in documentos_ordenados]

# =========================
# 8. PIPELINE RAG COMPLETO
# =========================

def responder_pergunta(pergunta, vectorstore):
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8
    )

    if not documentos_recuperados:
        return "Não encontrei informações relevantes nos documentos.", []

    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )

    contexto_final = documentos_rerankeados[:4]

    contexto_texto = "\n\n".join(
        [doc.page_content for doc in contexto_final]
    )

    prompt_final = f"""
Você é um agente de RH corporativo.
Responda APENAS com base nas políticas internas abaixo.

Contexto:
{contexto_texto}

Pergunta:
{pergunta}
"""

    resposta = llm.invoke(prompt_final)

    return resposta.content, contexto_final

# =========================
# 9. INTERFACE DE TERMINAL
# =========================

def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def imprimir_cabecalho():
    console.print(Panel.fit(
        "[bold blue]🤖 AGENTE DE RH — POLÍTICAS INTERNAS[/bold blue]\n"
        "[dim]RAG + Reranking com LangChain[/dim]",
        border_style="blue",
        padding=(1, 2)
    ))
    console.print("\nDigite sua pergunta ou '[bold]sair[/bold]' para encerrar.\n")

def imprimir_fontes(fontes):
    console.print(Panel(
        "[bold]📚 FONTES UTILIZADAS[/bold]",
        border_style="yellow",
        padding=(0, 1)
    ))
    
    for i, doc in enumerate(fontes, start=1):
        console.print(f"\n[bold cyan]Trecho {i}[/bold cyan]")
        console.print(f"  [dim]Documento:[/dim] {doc.metadata.get('documento', 'Desconhecido')}")
        console.print(f"  [dim]Categoria:[/dim] {doc.metadata.get('categoria', 'Geral')}")
        
        syntax = Syntax(
            doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else ""),
            "text",
            theme="monokai",
            line_numbers=False,
            word_wrap=True
        )
        console.print(syntax)

def main():
    limpar_tela()
    imprimir_cabecalho()

    try:
        vectorstore = inicializar_vectorstore()
    except Exception as e:
        console.print(Panel(f"[bold red]ERRO CRÍTICO:[/bold red] {e}", border_style="red"))
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
                    resposta, fontes = responder_pergunta(pergunta, vectorstore)
                except Exception as e:
                    console.print(f"\n[bold red]❌ Erro ao processar a pergunta:[/bold red] {e}")
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