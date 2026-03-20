# Instruções

## Dependências

Crie um ambiente virtual

```bash
uv venv
```

Instale as dependências

```bash
uv pip install -r requirements.txt
```

Crie um arquivo .env com a chave OPENAI_API_KEY

## Execução

Você pode executar um exemplo via web ou um exemplo via terminal.

### Para UI Web

Neste projeto precisa executar o próprio servidor web do Streamlit para subir aplicação

```bash
uv run streamlit run exemplos/langchain/main_web.py
```

### Para UI Terminal

Caso queira executar a aplicação em modo terminal (CLI) execute o seguinte comando:

Sem usar o framework Langchain

```bash
uv run exemplos/nativo/main_cli2_nativo.py
```

## Detalhes do Projeto

Disponível em [projeto.md](https://github.com/armandossrecife/my-rag-rh/blob/main/docs/projeto.md)
