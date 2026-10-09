"""Dashboard KIDY — somente retenção por coleções e Grupo Cliente."""
from datetime import date
from pathlib import Path
import pandas as pd
import streamlit as st
from grupos_cliente import adicionar_grupo
from painel_colecoes import mostrar

st.set_page_config(page_title='KIDY | Retenção por Coleção', page_icon='🎯', layout='wide')
SAIDA = Path(__file__).resolve().parent / 'saida'

@st.cache_data(show_spinner='Carregando carteira comercial…')
def carregar():
    ped = pd.read_parquet(SAIDA / 'pedidos.parquet')
    cli = pd.read_parquet(SAIDA / 'dim_clientes.parquet').set_index('cod_cliente')
    rep = pd.read_parquet(SAIDA / 'dim_representantes.parquet').set_index('cod_rep')['nome_rep']
    sup = pd.read_parquet(SAIDA / 'dim_supervisores.parquet').set_index('cod_sup')['nome_sup']
    ped = ped.merge(cli[['uf']], left_on='cod_cliente', right_index=True, how='left')
    return adicionar_grupo(ped, cli), cli, rep, sup

if not (SAIDA / 'pedidos.parquet').exists():
    st.error('Bases não encontradas. Execute atualizar_bases.bat antes de abrir o dashboard.')
    st.stop()

pedidos, clientes, reps, sups = carregar()
hoje = date.today()
referencia = min(hoje, max(hoje, pedidos['data'].max().date()))
st.sidebar.title('🎯 Retenção KIDY')
st.sidebar.caption('Carteira: grupos com compra desde 01/01/2024')
st.sidebar.caption('Churn: duas coleções encerradas sem compra; risco: uma coleção')
base = pedidos.loc[pedidos['cod_sup'] != 9900].copy()
sup_sel = st.sidebar.multiselect('Supervisor', sorted(base['cod_sup'].dropna().unique()), format_func=lambda x: f'{int(x)} - {sups.get(x, "")}' )
if sup_sel:
    base = base[base['cod_sup'].isin(sup_sel)]
rep_sel = st.sidebar.multiselect('Representante', sorted(base['cod_rep'].dropna().unique()), format_func=lambda x: f'{int(x)} - {reps.get(x, "")}' )
if rep_sel:
    base = base[base['cod_rep'].isin(rep_sel)]
uf_sel = st.sidebar.multiselect('UF', sorted(base['uf'].dropna().unique()))
if uf_sel:
    base = base[base['uf'].isin(uf_sel)]
st.title('Análise de Churn por Grupo Cliente')
st.caption(f'Carteira com compras desde 01/01/2024 · referência {referencia:%d/%m/%Y} · somente Grupo Cliente')
mostrar(base, clientes, referencia, reps, sups)
