# ============================================
# AGENTE DE RH COM RAG + RERANKING (VERSÃO CLI)
# LangChain + Terminal
# ============================================

# TODO: corrigir 🤖 Agente: Não encontrei informações relevantes nos documentos.   

# =========================
# 1. IMPORTAÇÕES
# =========================

import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.markdown import Markdown

# Cria uma instância do console para saída formatada
console = Console()

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Injeta a chave como variável de ambiente
if not os.getenv("OPENAI_API_KEY"):
    print("ERRO: OPENAI_API_KEY não encontrada no arquivo .env")
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
    """
    Carrega os PDFs de políticas internas de RH
    """
    caminhos = [
        "documentos/politica_ferias.pdf",
        "documentos/politica_home_office.pdf",
        "documentos/codigo_conduta.pdf"
    ]

    documentos = []

    console.print(">> Carregando documentos PDF...", end=" ")
    for caminho in caminhos:
        if not os.path.exists(caminho):
            console.print(f"\nAVISO: Arquivo não encontrado: {caminho}")
            continue
        loader = PyPDFLoader(caminho)
        docs = loader.load()

        for doc in docs:
            doc.metadata["documento"] = caminho

        documentos.extend(docs)
    
    console.print(f"OK ({len(documentos)} páginas carregadas)")
    return documentos

# =========================
# 4. CHUNKING
# =========================

def gerar_chunks(documentos):
    """
    Divide os documentos em chunks semânticos
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    return splitter.split_documents(documentos)

# =========================
# 5. ENRIQUECIMENTO COM METADADOS
# =========================

def enriquecer_chunks(chunks):
    """
    Classifica os chunks por categoria semântica
    """
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
    """
    Cria ou carrega o banco vetorial.
    Verifica persistência para evitar reprocessamento desnecessário.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Verifica se já existe um banco persistido
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        console.print(">> Banco vetorial existente detectado. Carregando...", end=" ")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        console.print("OK")
    else:
        console.print(">> Banco vetorial não encontrado. Processando documentos...")
        documentos = carregar_documentos()
        if not documentos:
            console.print("ERRO: Nenhum documento carregado. Verifique a pasta 'documentos'.")
            sys.exit(1)
            
        chunks = gerar_chunks(documentos)
        chunks = enriquecer_chunks(chunks)
        
        console.print(">> Criando embeddings e salvando banco...", end=" ")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        console.print("OK")

    return vectorstore

# =========================
# 7. RERANKING
# =========================

def rerank_documentos(pergunta, documentos, llm):
    """
    Reordena os documentos recuperados com base na relevância
    usando o próprio LLM (reranking semântico)
    """
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

    # Barra de progresso simples no terminal
    console.print(">> Realizando Reranking...", end=" ")
    
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

    # Ordena do mais relevante para o menos relevante
    documentos_ordenados = sorted(
        documentos_com_score,
        key=lambda x: x[0],
        reverse=True
    )

    console.print("OK")
    # Retorna apenas os documentos
    return [doc for _, doc in documentos_ordenados]

# =========================
# 8. PIPELINE RAG COMPLETO
# =========================

def responder_pergunta(pergunta, vectorstore):
    """
    Pipeline completo:
    - Recuperação
    - Reranking
    - Geração de resposta
    """
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0
    )

    # Recuperação inicial (top-k mais alto)
    documentos_recuperados = vectorstore.similarity_search(
        pergunta,
        k=8
    )

    if not documentos_recuperados:
        return "Não encontrei informações relevantes nos documentos.", []

    # Reranking
    documentos_rerankeados = rerank_documentos(
        pergunta,
        documentos_recuperados,
        llm
    )

    # Seleciona os melhores
    contexto_final = documentos_rerankeados[:4]

    # Prompt final
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
    console.print("=" * 60)
    console.print("🤖  AGENTE DE RH — POLÍTICAS INTERNAS (RAG + RERANKING)")
    console.print("=" * 60)
    console.print("Digite sua pergunta ou 'sair' para encerrar.\n")

def imprimir_fontes(fontes):
    console.print("\n" + "-" * 40)
    console.print("📚 FONTES UTILIZADAS:")
    console.print("-" * 40)
    for i, doc in enumerate(fontes, start=1):
        console.print(f"\n[Trecho {i}]")
        console.print(f"  Documento : {doc.metadata.get('documento', 'Desconhecido')}")
        console.print(f"  Categoria : {doc.metadata.get('categoria', 'Geral')}")
        console.print(f"  Conteúdo  : {doc.page_content[:150]}...") # Mostra apenas início para não poluir
    console.print("-" * 40)

def main():
    limpar_tela()
    imprimir_cabecalho()

    # Inicializa o Vector Store (Carrega ou Cria)
    try:
        vectorstore = inicializar_vectorstore()
    except Exception as e:
        console.print(f"\nERRO CRÍTICO ao inicializar banco de dados: {e}")
        sys.exit(1)

    console.print("\n✅ Sistema pronto para consultas.\n")

    while True:
        try:
            pergunta = input("👤 Você: ").strip()

            if pergunta.lower() in ["sair", "exit", "quit"]:
                console.print("\n👋 Encerrando agente de RH. Até logo!")
                break
            
            if not pergunta:
                continue

            console.print("\n⏳ Consultando políticas internas...")
            
            try:
                resposta, fontes = responder_pergunta(pergunta, vectorstore)
                
                console.print("\n🤖 Agente:")
                markdown_text = resposta
                markdown = Markdown(markdown_text, code_theme="monokai")
                console.print(markdown)
                
                if fontes:
                    imprimir_fontes(fontes)
                else:
                    console.print("\n⚠️  Nenhuma fonte específica foi utilizada para esta resposta.")
                    
            except Exception as e:
                console.print(f"\n❌ Erro ao processar a pergunta: {e}")

            console.print("\n" + "=" * 60 + "\n")

        except KeyboardInterrupt:
            console.print("\n\n👋 Interrupção detectada. Encerrando...")
            break

if __name__ == "__main__":
    main()