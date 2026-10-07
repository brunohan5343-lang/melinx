import io
import json
import os
import pandas as pd
import requests
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
import streamlit as st
from streamlit_option_menu import option_menu

st.set_page_config(page_title="MELInx", layout="wide")

# Configuração segura da chave de API do Gemini (Secrets do Streamlit Cloud ou Variável de Ambiente)
if "GOOGLE_API_KEY" in st.secrets:
    api_key_gemini = st.secrets["GOOGLE_API_KEY"]
else:
    api_key_gemini = os.getenv("GOOGLE_API_KEY")

modelo = OpenAI(
    api_key=api_key_gemini,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
)

CSV_FILE = "produtos_meli.csv"
CSV_ORIGINAL_ML = "produtos_meli_original.csv"


def padronizar_colunas(df):
    if df.empty:
        return df
    df.columns = [str(c).strip() for c in df.columns]
    return df


def carregar_dados():
    try:
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
            return padronizar_colunas(df)
        return pd.DataFrame(
            columns=[
                "ID do Anúncio",
                "Título do Produto",
                "Preço (R$)",
                "Estoque Disponível",
                "Status",
                "Condição",
                "Link",
            ]
        )
    except Exception:
        return pd.DataFrame(
            columns=[
                "ID do Anúncio",
                "Título do Produto",
                "Preço (R$)",
                "Estoque Disponível",
                "Status",
                "Condição",
                "Link",
            ]
        )


def carregar_dados_originais():
    try:
        if os.path.exists(CSV_ORIGINAL_ML):
            df = pd.read_csv(CSV_ORIGINAL_ML)
            return padronizar_colunas(df)
        else:
            df = carregar_dados()
            df.to_csv(CSV_ORIGINAL_ML, index=False)
            return df
    except Exception:
        return pd.DataFrame(
            columns=[
                "ID do Anúncio",
                "Título do Produto",
                "Preço (R$)",
                "Estoque Disponível",
                "Status",
                "Condição",
                "Link",
            ]
        )


if "mensagem_feedback" not in st.session_state:
    st.session_state["mensagem_feedback"] = None


def sincronizar_com_ml(access_token):
    headers = {"Authorization": f"Bearer {access_token}"}

    resp_user = requests.get(
        "https://api.mercadolibre.com/users/me", headers=headers
    )
    if resp_user.status_code != 200:
        return (
            False,
            "Token inválido ou expirado. Verifique a credencial informada.",
        )

    seller_id = resp_user.json().get("id")

    resp_itens = requests.get(
        f"https://api.mercadolibre.com/users/{seller_id}/items/search",
        headers=headers,
    )
    if resp_itens.status_code != 200:
        return False, "Erro ao buscar os itens da conta."

    item_ids = resp_itens.json().get("results", [])
    produtos_api = []

    for item_id in item_ids:
        resp_detalhe = requests.get(
            f"https://api.mercadolibre.com/items/{item_id}", headers=headers
        )
        if resp_detalhe.status_code == 200:
            item = resp_detalhe.json()
            produtos_api.append({
                "ID do Anúncio": str(item.get("id")),
                "Título do Produto": str(item.get("title")),
                "Preço (R$)": float(item.get("price", 0.0)),
                "Estoque Disponível": int(item.get("available_quantity", 0)),
                "Status": str(item.get("status")),
                "Condição": str(item.get("condition")),
                "Link": str(item.get("permalink")),
            })

    if produtos_api:
        df_novo = pd.DataFrame(produtos_api)
        df_novo = padronizar_colunas(df_novo)
        df_novo.to_csv(CSV_FILE, index=False)
        df_novo.to_csv(CSV_ORIGINAL_ML, index=False)
        st.session_state["tabela_produtos"] = df_novo
        return (
            True,
            f"{len(produtos_api)} produtos sincronizados com sucesso do Mercado"
            " Livre!",
        )

    return False, "Nenhum produto encontrado nesta conta do Mercado Livre."


def cadastrar_produto_local(
    id_anuncio,
    titulo,
    preco,
    estoque,
    status="active",
    condicao="new",
    link="https://produto.mercadolivre.com.br/",
):
    df_atual = carregar_dados()
    id_str = str(id_anuncio).strip()

    if not df_atual.empty and "ID do Anúncio" in df_atual.columns:
        if id_str in df_atual["ID do Anúncio"].astype(str).values:
            return False, f"Já existe um produto com o ID {id_str} cadastrado."

    novo_produto = {
        "ID do Anúncio": id_str,
        "Título do Produto": str(titulo).strip(),
        "Preço (R$)": float(preco),
        "Estoque Disponível": int(estoque),
        "Status": str(status),
        "Condição": str(condicao),
        "Link": str(link),
    }
    novo_df = pd.DataFrame([novo_produto])
    df_atual = pd.concat([df_atual, novo_df], ignore_index=True)
    df_atual.to_csv(CSV_FILE, index=False)
    st.session_state["tabela_produtos"] = df_atual
    return True, f"Produto '{titulo}' (ID: {id_str}) cadastrado com sucesso!"


def editar_produto_local(
    termo_busca, novo_preco=None, novo_estoque=None, novo_status=None
):
    df_atual = carregar_dados()
    if df_atual.empty:
        return False, "Não há produtos cadastrados."

    mask = (
        df_atual["ID do Anúncio"]
        .astype(str)
        .str.contains(termo_busca, case=False, na=False)
    ) | (
        df_atual["Título do Produto"]
        .astype(str)
        .str.contains(termo_busca, case=False, na=False)
    )

    if not mask.any():
        return False, f"Nenhum produto encontrado com o termo '{termo_busca}'."

    indices = df_atual[mask].index
    alterados = []

    for idx in indices:
        if novo_preco is not None:
            df_atual.at[idx, "Preço (R$)"] = float(novo_preco)
        if novo_estoque is not None:
            df_atual.at[idx, "Estoque Disponível"] = int(novo_estoque)
        if novo_status is not None:
            df_atual.at[idx, "Status"] = str(novo_status)
        alterados.append(df_atual.at[idx, "Título do Produto"])

    if "Opcao_Exibicao" in df_atual.columns:
        df_atual = df_atual.drop(columns=["Opcao_Exibicao"])

    df_atual.to_csv(CSV_FILE, index=False)
    st.session_state["tabela_produtos"] = df_atual
    nomes_alterados = ", ".join(alterados)
    return True, f"Produto(s) atualizado(s) com sucesso: {nomes_alterados}"


def remover_produto_local(termo_busca):
    df_atual = carregar_dados()
    if df_atual.empty:
        return False, "Não há produtos cadastrados."

    mask = (
        df_atual["ID do Anúncio"]
        .astype(str)
        .str.contains(termo_busca, case=False, na=False)
    ) | (
        df_atual["Título do Produto"]
        .astype(str)
        .str.contains(termo_busca, case=False, na=False)
    )

    if not mask.any():
        return False, f"Nenhum produto encontrado com o termo '{termo_busca}'."

    removidos = df_atual[mask]["Título do Produto"].tolist()
    df_atual = df_atual[~mask]

    if "Opcao_Exibicao" in df_atual.columns:
        df_atual = df_atual.drop(columns=["Opcao_Exibicao"])

    df_atual.to_csv(CSV_FILE, index=False)
    st.session_state["tabela_produtos"] = df_atual
    nomes_removidos = ", ".join(removidos)
    return True, f"Produto(s) removido(s): {nomes_removidos}"


if "tabela_produtos" not in st.session_state:
    st.session_state["tabela_produtos"] = carregar_dados()

with st.sidebar:
    menu = option_menu(
        menu_title="MELInx",
        options=[
            "Produtos",
            "Cadastrar Produto",
            "Editar Produto",
            "Relatório",
            "Assistente de IA",
        ],
        icons=[
            "table",
            "cloud-arrow-down-fill",
            "pencil-square",
            "file-earmark-text-fill",
            "chat-dots-fill",
        ],
        menu_icon="menu-app-filled",
        default_index=0,
        styles={
            "container": {
                "padding": "5!important",
                "background-color": "transparent",
            },
            "icon": {"color": "#4da6ff", "font-size": "18px"},
            "nav-link": {
                "font-size": "15px",
                "text-align": "left",
                "margin": "5px 0px",
                "border-radius": "8px",
                "--hover-color": "#1e3d59",
            },
            "nav-link-selected": {
                "background-color": "#007acc",
                "color": "white",
                "font-weight": "600",
                "box-shadow": "0 0 12px rgba(0, 122, 204, 0.7)",
            },
        },
    )

    st.markdown("---")
    st.markdown("### ⚙️ Configuração da Conta")

    token_input = st.text_input(
        "Token de Acesso do Mercado Livre",
        value=st.session_state.get("ml_token", ""),
        type="password",
        placeholder="Cole seu APP_USR-... aqui",
    )

    if token_input != st.session_state.get("ml_token", ""):
        st.session_state["ml_token"] = token_input
        if token_input:
            st.success("Token atualizado com sucesso!")
        else:
            st.warning("Token removido.")

    if st.button(
        "🔄 Sincronizar com o Mercado Livre Agora", use_container_width=True
    ):
        token_atual = st.session_state.get("ml_token", "")
        if not token_atual:
            st.warning(
                "Insira o seu Token de Acesso do Mercado Livre acima antes de"
                " sincronizar."
            )
        else:
            with st.spinner("Conectando à API do Mercado Livre..."):
                sucesso, mensagem = sincronizar_com_ml(token_atual)
                if sucesso:
                    st.success(mensagem)
                    st.rerun()
                else:
                    st.error(mensagem)

if menu == "Produtos":
    col_img, col_titulo = st.columns([0.24, 0.76])
    with col_img:
        try:
            st.image("imagem.jpg", width=220)
        except Exception:
            pass
    with col_titulo:
        st.markdown(
            "<h2 style='margin-top: 30px; font-size: 34px;'>MELInx - Sistema de"
            " Integração Local</h2>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    df_atual = carregar_dados()
    st.dataframe(df_atual, use_container_width=True)

elif menu == "Cadastrar Produto":
    st.title("➕ Cadastrar Novo Produto Localmente")

    col1, col2 = st.columns(2)
    with col1:
        id_anuncio = st.text_input("ID do Anúncio (ex: MLB1234567890)")
        titulo = st.text_input("Título do Produto")
        preco = st.number_input("Preço (R$)", min_value=0.0, format="%.2f")
    with col2:
        estoque = st.number_input("Estoque Disponível", min_value=0, step=1)
        status = st.selectbox("Status", ["active", "paused"])
        condicao = st.selectbox("Condição", ["new", "used"])
        link = st.text_input(
            "Link do Anúncio", value="https://produto.mercadolivre.com.br/"
        )

    botao_cadastrar = st.button("Cadastrar Produto")

    if botao_cadastrar:
        if id_anuncio and titulo:
            ok, msg = cadastrar_produto_local(
                id_anuncio, titulo, preco, estoque, status, condicao, link
            )
            if ok:
                st.success(msg)
            else:
                st.error(msg)
        else:
            st.warning("Preencha pelo menos o ID do Anúncio e o Título do Produto.")

elif menu == "Editar Produto":
    st.title("✏️ Editar ou Remover Produto (Local)")

    if st.session_state["mensagem_feedback"]:
        tipo, texto = st.session_state["mensagem_feedback"]
        if tipo == "success":
            st.success(texto)
        elif tipo == "warning":
            st.warning(texto)
        elif tipo == "error":
            st.error(texto)
        st.session_state["mensagem_feedback"] = None

    df_atual = carregar_dados()

    if df_atual.empty:
        st.info("Não há produtos cadastrados para editar.")
    else:
        df_atual["Opcao_Exibicao"] = (
            df_atual["Título do Produto"].astype(str)
            + " (ID: "
            + df_atual["ID do Anúncio"].astype(str)
            + ")"
        )

        lista_opcoes = list(df_atual["Opcao_Exibicao"])
        produto_escolhido = st.selectbox(
            "Selecione o produto",
            options=lista_opcoes,
            index=None,
            placeholder="Selecione o produto pelo Nome ou ID...",
        )

        if produto_escolhido:
            id_para_gerenciar = df_atual[
                df_atual["Opcao_Exibicao"] == produto_escolhido
            ]["ID do Anúncio"].values[0]
            prod_selecionado = df_atual[
                df_atual["ID do Anúncio"] == id_para_gerenciar
            ].iloc[0]

            nome_prod = prod_selecionado["Título do Produto"]
            id_prod = prod_selecionado["ID do Anúncio"]

            with st.form("form_edicao"):
                st.write(f"**Produto Selecionado:** {nome_prod}")
                st.write(f"**ID do Anúncio:** {id_prod}")

                novo_preco = st.number_input(
                    "Preço (R$)",
                    value=float(prod_selecionado["Preço (R$)"]),
                    format="%.2f",
                )
                novo_estoque = st.number_input(
                    "Estoque Disponível",
                    value=int(prod_selecionado["Estoque Disponível"]),
                    step=1,
                )
                novo_status = st.selectbox(
                    "Status",
                    ["active", "paused"],
                    index=0 if prod_selecionado["Status"] == "active" else 1,
                )

                col_b1, col_b2 = st.columns(2)
                btn_atualizar = col_b1.form_submit_button("Atualizar Dados")
                btn_remover = col_b2.form_submit_button("Remover Produto")

                if btn_atualizar:
                    df_atual.loc[
                        df_atual["ID do Anúncio"].astype(str)
                        == str(id_para_gerenciar),
                        "Preço (R$)",
                    ] = float(novo_preco)
                    df_atual.loc[
                        df_atual["ID do Anúncio"].astype(str)
                        == str(id_para_gerenciar),
                        "Estoque Disponível",
                    ] = int(novo_estoque)
                    df_atual.loc[
                        df_atual["ID do Anúncio"].astype(str)
                        == str(id_para_gerenciar),
                        "Status",
                    ] = str(novo_status)
                    if "Opcao_Exibicao" in df_atual.columns:
                        df_atual = df_atual.drop(columns=["Opcao_Exibicao"])
                    df_atual.to_csv(CSV_FILE, index=False)
                    st.session_state["tabela_produtos"] = df_atual

                    st.session_state["mensagem_feedback"] = (
                        "success",
                        "Produto atualizado localmente com sucesso!",
                    )
                    st.rerun()

                if btn_remover:
                    df_atual = df_atual[
                        df_atual["ID do Anúncio"].astype(str)
                        != str(id_para_gerenciar)
                    ]
                    if "Opcao_Exibicao" in df_atual.columns:
                        df_atual = df_atual.drop(columns=["Opcao_Exibicao"])
                    df_atual.to_csv(CSV_FILE, index=False)
                    st.session_state["tabela_produtos"] = df_atual
                    st.session_state["mensagem_feedback"] = (
                        "success",
                        "Produto removido localmente com sucesso!",
                    )
                    st.rerun()

elif menu == "Relatório":
    st.title("📊 Relatório de Alterações Locais")
    st.info(
        "Este relatório compara o estado atual dos seus produtos locais com a"
        " última sincronização realizada direto do Mercado Livre."
    )

    df_original = carregar_dados_originais()
    df_atual = carregar_dados()

    if df_original.empty or "ID do Anúncio" not in df_original.columns:
        st.warning(
            "Nenhum dado original de sincronização encontrado. Faça uma"
            " sincronização com o Mercado Livre para gerar a base de comparação."
        )
    else:
        try:
            ids_originais = set(df_original["ID do Anúncio"].astype(str))
            ids_atuais = set(df_atual["ID do Anúncio"].astype(str))

            ids_adicionados = ids_atuais - ids_originais
            df_adicionados = df_atual[
                df_atual["ID do Anúncio"].astype(str).isin(ids_adicionados)
            ]

            ids_excluidos = ids_originais - ids_atuais
            df_excluidos = df_original[
                df_original["ID do Anúncio"].astype(str).isin(ids_excluidos)
            ]

            ids_comuns = ids_originais.intersection(ids_atuais)
            alteracoes = []

            col_preco = next(
                (
                    c
                    for c in ["Preço (R$)", "Preco", "Preço"]
                    if c in df_atual.columns
                ),
                None,
            )
            col_estoque = next(
                (
                    c
                    for c in [
                        "Estoque Disponível",
                        "Estoque",
                        "available_quantity",
                    ]
                    if c in df_atual.columns
                ),
                None,
            )
            col_status = next(
                (c for c in ["Status", "status"] if c in df_atual.columns), None
            )
            col_titulo = next(
                (
                    c
                    for c in [
                        "Título do Produto",
                        "Titulo do Produto",
                        "title",
                    ]
                    if c in df_atual.columns
                ),
                "Título do Produto",
            )

            for aid in ids_comuns:
                orig = df_original[
                    df_original["ID do Anúncio"].astype(str) == aid
                ].iloc[0]
                atual = df_atual[
                    df_atual["ID do Anúncio"].astype(str) == aid
                ].iloc[0]

                mudancas = []

                if col_preco and col_preco in orig and col_preco in atual:
                    try:
                        p_orig = round(float(orig[col_preco]), 2)
                        p_atual = round(float(atual[col_preco]), 2)
                        if p_orig != p_atual:
                            mudancas.append(
                                f"Preço: R$ {p_orig:.2f} ➔ R$ {p_atual:.2f}"
                            )
                    except ValueError:
                        if str(orig[col_preco]) != str(atual[col_preco]):
                            mudancas.append(
                                f"Preço: {orig[col_preco]} ➔ {atual[col_preco]}"
                            )

                if (
                    col_estoque
                    and col_estoque in orig
                    and col_estoque in atual
                ):
                    try:
                        e_orig = int(float(orig[col_estoque]))
                        e_atual = int(float(atual[col_estoque]))
                        if e_orig != e_atual:
                            mudancas.append(
                                f"Estoque: {e_orig} ➔ {e_atual}"
                            )
                    except ValueError:
                        if str(orig[col_estoque]) != str(atual[col_estoque]):
                            mudancas.append(
                                f"Estoque: {orig[col_estoque]} ➔"
                                f" {atual[col_estoque]}"
                            )

                if col_status and col_status in orig and col_status in atual:
                    s_orig = str(orig[col_status]).strip().lower()
                    s_atual = str(atual[col_status]).strip().lower()
                    if s_orig != s_atual:
                        mudancas.append(
                            f"Status: {orig[col_status]} ➔ {atual[col_status]}"
                        )

                if mudancas:
                    alteracoes.append({
                        "ID do Anúncio": aid,
                        "Título do Produto": atual.get(col_titulo, "N/D"),
                        "Alterações Realizadas": " | ".join(mudancas),
                    })

            df_editados = pd.DataFrame(alteracoes)

            st.markdown("### 🟢 Produtos Adicionados Localmente")
            if not df_adicionados.empty:
                st.dataframe(df_adicionados, use_container_width=True)
            else:
                st.write("Nenhum produto novo adicionado.")

            st.markdown("### 🔴 Produtos Excluídos Localmente")
            if not df_excluidos.empty:
                st.dataframe(df_excluidos, use_container_width=True)
            else:
                st.write("Nenhum produto excluído.")

            st.markdown("### 🟡 Produtos Editados Localmente")
            if not df_editados.empty:
                st.dataframe(df_editados, use_container_width=True)
            else:
                st.write("Nenhum produto editado.")

            def gerar_pdf_relatorio():
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(
                    buffer,
                    pagesize=letter,
                    rightMargin=30,
                    leftMargin=30,
                    topMargin=30,
                    bottomMargin=30,
                )
                elementos = []
                estilos = getSampleStyleSheet()

                estilo_titulo = ParagraphStyle(
                    "TituloRelatorio",
                    parent=estilos["Heading1"],
                    fontSize=18,
                    textColor=colors.HexColor("#007acc"),
                    spaceAfter=15,
                )

                estilo_sub = ParagraphStyle(
                    "SubRelatorio",
                    parent=estilos["Heading2"],
                    fontSize=14,
                    textColor=colors.HexColor("#1e3d59"),
                    spaceBefore=12,
                    spaceAfter=8,
                )

                estilo_celula = ParagraphStyle(
                    "CelulaTabela",
                    parent=estilos["Normal"],
                    fontSize=8,
                    leading=10,
                    textColor=colors.HexColor("#333333"),
                )

                estilo_cabecalho = ParagraphStyle(
                    "CabecalhoTabela",
                    parent=estilos["Normal"],
                    fontSize=9,
                    leading=11,
                    textColor=colors.whitesmoke,
                    fontName="Helvetica-Bold",
                )

                elementos.append(
                    Paragraph(
                        "Relatório de Alterações Locais - MELInx", estilo_titulo
                    )
                )

                def dataframe_para_tabela(df_in):
                    if df_in.empty:
                        return Paragraph("Nenhum registro.", estilos["Normal"])

                    dados = []
                    cabecalho = [
                        Paragraph(str(col), estilo_cabecalho)
                        for col in df_in.columns
                    ]
                    dados.append(cabecalho)

                    for _, linha in df_in.iterrows():
                        linha_paragrafos = [
                            Paragraph(str(val), estilo_celula)
                            for val in linha.values
                        ]
                        dados.append(linha_paragrafos)

                    tabela = Table(dados, colWidths=[90, 160, 302])
                    tabela.setStyle(
                        TableStyle([
                            (
                                "BACKGROUND",
                                (0, 0),
                                (-1, 0),
                                colors.HexColor("#007acc"),
                            ),
                            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                            ("TOPPADDING", (0, 0), (-1, -1), 5),
                            (
                                "BACKGROUND",
                                (0, 1),
                                (-1, -1),
                                colors.HexColor("#f2f2f2"),
                            ),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ])
                    )
                    return tabela

                elementos.append(
                    Paragraph("Produtos Adicionados Localmente", estilo_sub)
                )
                if not df_adicionados.empty:
                    cols_add = [
                        c
                        for c in [
                            "ID do Anúncio",
                            "Título do Produto",
                            "Preço (R$)",
                            "Estoque Disponível",
                        ]
                        if c in df_adicionados.columns
                    ]
                    elementos.append(
                        dataframe_para_tabela(df_adicionados[cols_add])
                    )
                else:
                    elementos.append(
                        Paragraph(
                            "Nenhum produto novo adicionado.",
                            estilos["Normal"],
                        )
                    )

                elementos.append(
                    Paragraph("Produtos Excluídos Localmente", estilo_sub)
                )
                if not df_excluidos.empty:
                    cols_exc = [
                        c
                        for c in [
                            "ID do Anúncio",
                            "Título do Produto",
                            "Preço (R$)",
                            "Estoque Disponível",
                        ]
                        if c in df_excluidos.columns
                    ]
                    elementos.append(
                        dataframe_para_tabela(df_excluidos[cols_exc])
                    )
                else:
                    elementos.append(
                        Paragraph(
                            "Nenhum produto excluído.", estilos["Normal"]
                        )
                    )

                elementos.append(
                    Paragraph("Produtos Editados Localmente", estilo_sub)
                )
                if not df_editados.empty:
                    elementos.append(dataframe_para_tabela(df_editados))
                else:
                    elementos.append(
                        Paragraph("Nenhum produto editado.", estilos["Normal"])
                    )

                doc.build(elementos)
                buffer.seek(0)
                return buffer.getvalue()

            st.markdown("<br>", unsafe_allow_html=True)
            pdf_bytes = gerar_pdf_relatorio()
            st.download_button(
                label="📥 Exportar Relatório em PDF",
                data=pdf_bytes,
                file_name="relatorio_alteracoes_melinx.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        except Exception as ex:
            st.error(
                f"Erro ao gerar o relatório: {ex}. Tente fazer uma nova"
                " sincronização com o Mercado Livre."
            )

elif menu == "Assistente de IA":
    st.title("💬 Assistente de IA")
    st.info(
        "Converse com a IA para consultar, cadastrar, editar ou remover"
        " produtos localmente."
    )

    if not "lista_mensagens" in st.session_state:
        st.session_state["lista_mensagens"] = []

    for mensagem in st.session_state["lista_mensagens"]:
        if mensagem["role"] != "system":
            st.chat_message(mensagem["role"]).write(mensagem["content"])

    mensagem_usuario = st.chat_input("Digite sua mensagem aqui...")

    if mensagem_usuario:
        st.chat_message("user").write(mensagem_usuario)
        st.session_state["lista_mensagens"].append(
            {"role": "user", "content": mensagem_usuario}
        )

        df_atual = carregar_dados()
        csv_str = df_atual.to_csv(index=False)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "cadastrar_produto_assistente",
                    "description": (
                        "Cadastra um novo produto na base de dados local do"
                        " sistema."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "id_anuncio": {
                                "type": "string",
                                "description": (
                                    "O ID do anúncio, ex: MLB123456789"
                                ),
                            },
                            "titulo": {
                                "type": "string",
                                "description": (
                                    "O título ou nome do produto"
                                ),
                            },
                            "preco": {
                                "type": "number",
                                "description": (
                                    "O preço do produto em reais"
                                ),
                            },
                            "estoque": {
                                "type": "integer",
                                "description": (
                                    "A quantidade disponível em estoque"
                                ),
                            },
                            "status": {
                                "type": "string",
                                "description": (
                                    "Status do produto ('active' ou 'paused')"
                                ),
                                "default": "active",
                            },
                            "condicao": {
                                "type": "string",
                                "description": (
                                    "Condição do produto ('new' ou 'used')"
                                ),
                                "default": "new",
                            },
                            "link": {
                                "type": "string",
                                "description": "Link do anúncio",
                                "default": (
                                    "https://produto.mercadolivre.com.br/"
                                ),
                            },
                        },
                        "required": [
                            "id_anuncio",
                            "titulo",
                            "preco",
                            "estoque",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "editar_produto_assistente",
                    "description": (
                        "Edita preço, estoque ou status de um produto existente"
                        " buscando pelo ID ou parte do título."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "termo_busca": {
                                "type": "string",
                                "description": (
                                    "O ID ou parte do nome do produto a ser"
                                    " alterado"
                                ),
                            },
                            "novo_preco": {
                                "type": "number",
                                "description": (
                                    "Novo preço do produto (opcional)"
                                ),
                            },
                            "novo_estoque": {
                                "type": "integer",
                                "description": (
                                    "Novo estoque disponível (opcional)"
                                ),
                            },
                            "novo_status": {
                                "type": "string",
                                "description": (
                                    "Novo status ('active' ou 'paused')"
                                    " (opcional)"
                                ),
                            },
                        },
                        "required": ["termo_busca"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remover_produto_assistente",
                    "description": (
                        "Remove um produto da base de dados local buscando pelo"
                        " ID ou parte do título."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "termo_busca": {
                                "type": "string",
                                "description": (
                                    "O ID ou parte do nome do produto a ser"
                                    " removido"
                                ),
                            }
                        },
                        "required": ["termo_busca"],
                    },
                },
            },
        ]

        system_prompt = {
            "role": "system",
            "content": (
                "Você é um assistente de IA especialista em gerenciar produtos"
                f" locais. Aqui está o conteúdo atual do arquivo 'produtos_meli.csv':\n\n{csv_str}\n\nQuando"
                " o usuário pedir para cadastrar, editar ou remover um produto,"
                " chame a função correspondente.\n\n"
                "IMPORTANTE: Ao redigir respostas, NUNCA coloque asteriscos (*)"
                " colados em símbolos de moeda (como R$) ou números. Garanta"
                " sempre que haja um espaço adequado para evitar erros de"
                " formatação na tela."
            ),
        }

        mensagens_for_envio = [system_prompt] + [
            m for m in st.session_state["lista_mensagens"] if m["role"] != "system"
        ]

        try:
            resposta_modelo = modelo.chat.completions.create(
                messages=mensagens_for_envio,
                model="gemini-flash-lite-latest",
                tools=tools,
                tool_choice="auto",
            )

            response_message = resposta_modelo.choices[0].message

            if response_message.tool_calls:
                mensagens_for_envio.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(
                        tool_call.function.arguments
                    )

                    if function_name == "cadastrar_produto_assistente":
                        ok, msg_cad = cadastrar_produto_local(
                            id_anuncio=function_args.get("id_anuncio"),
                            titulo=function_args.get("titulo"),
                            preco=function_args.get("preco"),
                            estoque=function_args.get("estoque"),
                            status=function_args.get("status", "active"),
                            condicao=function_args.get("condicao", "new"),
                            link=function_args.get(
                                "link",
                                "https://produto.mercadolivre.com.br/",
                            ),
                        )
                        resultado_func = msg_cad
                    elif function_name == "editar_produto_assistente":
                        ok, msg_edit = editar_produto_local(
                            termo_busca=function_args.get("termo_busca"),
                            novo_preco=function_args.get("novo_preco"),
                            novo_estoque=function_args.get("novo_estoque"),
                            novo_status=function_args.get("novo_status"),
                        )
                        resultado_func = msg_edit
                    elif function_name == "remover_produto_assistente":
                        ok, msg_rem = remover_produto_local(
                            termo_busca=function_args.get("termo_busca")
                        )
                        resultado_func = msg_rem
                    else:
                        resultado_func = "Função desconhecida."

                    mensagens_for_envio.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": resultado_func,
                    })

                segunda_resposta = modelo.chat.completions.create(
                    messages=mensagens_for_envio,
                    model="gemini-flash-lite-latest",
                )
                resposta_ia = segunda_resposta.choices[0].message.content
            else:
                resposta_ia = response_message.content

            if resposta_ia:
                st.chat_message("assistant").write(resposta_ia)
                st.session_state["lista_mensagens"].append(
                    {"role": "assistant", "content": resposta_ia}
                )

        except Exception as e:
            st.error(f"Erro ao processar solicitação: {e}")
