# Instruções

## Dependências

```bash
uv pip install -r requirements.txt
```

## Execução

Neste projeto precisa executar o próprio servidor web do Streamlit para subir aplicação

```bash
uv run streamlit run exemplos/langchain/main_web.py
```

Caso queira executar a aplicação em modo terminal (CLI) execute o seguinte comando:

Sem usar o framework Langchain

```bash
uv run exemplos/nativo/main_cli2_nativo.py
```