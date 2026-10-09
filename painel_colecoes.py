"""Painel executivo de saúde da carteira por Grupo Cliente."""
import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from churn_colecoes import calcular_colecoes

DATA_INICIO = pd.Timestamp('2024-01-01')
ROTULOS = {'Ativo':'Continuam comprando','Possível churn':'Precisam de atenção','Churn':'Deixaram de comprar','Novo na última coleção':'Novos / sem histórico suficiente'}


def _num(n):
    return f'{int(n):,}'.replace(',', '.')


def mostrar(pedidos, clientes, referencia, reps, sups):
    dados, m = calcular_colecoes(pedidos, clientes, referencia)
    st.title('📊 Saúde da carteira de clientes')
    st.write('**Veja quem está comprando, quem precisa de atenção e quem devemos recuperar.**')
    st.caption(f'Última compra a partir de 01/01/2024 • Situação calculada até a coleção encerrada em {m["fim"]:%d/%m/%Y} • Um registro por Grupo Cliente')
    if dados.empty:
        st.warning('Nenhum grupo de clientes encontrado para os filtros escolhidos.')
        return
    dados = dados.loc[pd.to_datetime(dados['ultima_compra'], errors='coerce') >= DATA_INICIO].copy()
    eleg = dados.loc[dados['elegivel']].copy()
    total = len(eleg)
    quant = {s:int((eleg['status']==s).sum()) for s in ('Ativo','Possível churn','Churn')}
    recuperados = dados.loc[dados['reativado_atual'].fillna(False)]
    st.markdown('### 1. Como estão nossos clientes?')
    st.caption(f'Analisamos {_num(total)} grupos com histórico suficiente para avaliar a perda de coleções. Clique em **Ver clientes** para abrir os nomes.')
    cards = [
        ('Ativo','🟢','Continuam comprando','Não estão classificados como perda nas coleções encerradas.'),
        ('Possível churn','🟠','Precisam de atenção','Passaram uma coleção completa sem comprar.'),
        ('Churn','🔴','Deixaram de comprar','Passaram duas ou mais coleções completas sem comprar.')
    ]
    colunas = st.columns(3)
    for col,(status,icone,titulo,explicacao) in zip(colunas,cards):
        n=quant[status]
        with col:
            with st.container(border=True):
                st.markdown(f'**{icone} {titulo}**')
                st.metric('Grupos de clientes',_num(n))
                st.write(f'**{n/total:.1%}** dos grupos analisados' if total else '0%')
                st.caption(explicacao)
                if st.button('🔎 Ver clientes',key=f'abrir_{status}',use_container_width=True):
                    st.session_state['filtro_situacao_exec'] = status
    st.markdown('### 2. O que esses números significam?')
    em_atencao = quant['Possível churn']+quant['Churn']
    st.info(f'**Em cada 100 grupos analisados:** aproximadamente {round(100*quant["Ativo"]/total) if total else 0} continuam comprando, {round(100*quant["Possível churn"]/total) if total else 0} precisam de atenção e {round(100*quant["Churn"]/total) if total else 0} deixaram de comprar por pelo menos duas coleções completas.')
    st.warning(f'**Ação comercial:** {_num(em_atencao)} grupos merecem avaliação. Isso não significa que todo esse faturamento foi perdido; precisamos verificar a situação e o potencial de cada cliente.')
    with st.expander('Como classificamos os clientes?'):
        st.write('🟢 **Continuam comprando:** não ficaram uma coleção completa sem compra, conforme a regra histórica.')
        st.write('🟠 **Precisam de atenção:** uma coleção encerrada sem compra.')
        st.write('🔴 **Deixaram de comprar:** duas ou mais coleções encerradas consecutivas sem compra.')
        st.write('**Importante:** compras na coleção em andamento são mostradas separadamente. Um grupo pode ter voltado a comprar agora e ainda aparecer na classificação histórica de uma coleção anterior.')
    st.markdown('### 3. Situação financeira dos clientes em atenção ou perdidos')
    problem = eleg.loc[eleg['status'].isin(['Churn', 'Possível churn'])]
    n_problem = len(problem)
    bloqueados = problem.loc[problem['bloqueio'].isin(['Todas', 'Parcial'])]
    sem_limite = problem.loc[problem['situacao_limite'] == 'Sem limite']
    a, b = st.columns(2)
    for col, titulo, sub, df in [
        (a, '🟠 Sem limite positivo', 'Verificar crédito antes da negociação', sem_limite),
        (b, '🔴 Com algum bloqueio', 'Consultar a equipe financeira', bloqueados),
    ]:
        with col:
            with st.container(border=True):
                st.markdown(f'**{titulo}**')
                st.metric('Grupos', _num(len(df)))
                st.caption(f'{len(df)/n_problem:.1%} dos grupos em atenção/perdidos' if n_problem else 'Sem grupos')
                st.write(sub)
    st.caption('As situações financeiras podem se sobrepor. Bloqueio é observado em qualquer empresa do grupo.')
    st.markdown('### 4. Quais são os clientes?')
    st.write('Escolha uma situação e veja **os nomes dos grupos**, o representante, a última compra e o que fazer.')
    opcoes = list(ROTULOS)
    selecionada = st.selectbox('Mostrar clientes',opcoes,format_func=lambda x:ROTULOS[x],key='filtro_situacao_exec')
    busca = st.text_input('Pesquisar pelo nome do Grupo Cliente',placeholder='Digite parte do nome')
    tabela = dados.loc[dados['status']==selecionada].copy()
    if busca.strip():
        tabela = tabela.loc[tabela['nome'].astype(str).str.contains(busca.strip(),case=False,regex=False)]
    tabela['representante'] = tabela['cod_rep'].map(lambda x: f'{int(x)} - {reps.get(x, "")}' if pd.notna(x) else '')
    tabela['supervisor'] = tabela['cod_sup'].map(lambda x: f'{int(x)} - {sups.get(x, "")}' if pd.notna(x) else '')
    tabela['acao'] = tabela.apply(lambda r: ('Consultar financeiro' if r['bloqueio']!='Não' else 'Verificar limite' if r['situacao_limite']!='Com limite' else 'Entrar em contato') if r['status']!='Ativo' else 'Manter relacionamento',axis=1)
    tabela['compra_atual'] = tabela['compra_atual'].map({True:'Sim',False:'Não'})
    tabela['reativado_atual'] = tabela['reativado_atual'].map({True:'Sim',False:'Não'})
    colunas_tabela = {'nome':'Grupo Cliente','status':'Situação técnica','ultima_compra':'Última compra','ultima_colecao':'Última coleção comprada','representante':'Representante','supervisor':'Supervisor','colecoes_sem_compra':'Coleções sem compra','compra_atual':'Comprou na coleção atual?','reativado_atual':'Voltou a comprar agora?','bloqueio':'Bloqueio financeiro','situacao_limite':'Limite de crédito','limite_total':'Limite informado (R$)','faturamento_historico':'Compras históricas (R$)','acao':'Próxima ação'}
    tabela = tabela[list(colunas_tabela)].rename(columns=colunas_tabela).sort_values('Última compra',ascending=False)
    st.success(f'**{_num(len(tabela))} grupos encontrados** — {ROTULOS[selecionada]}.')
    st.dataframe(tabela,use_container_width=True,hide_index=True,column_config={'Última compra':st.column_config.DateColumn(format='DD/MM/YYYY'),'Limite informado (R$)':st.column_config.NumberColumn(format='R$ %.2f'),'Compras históricas (R$)':st.column_config.NumberColumn(format='R$ %.2f')})
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine='openpyxl') as writer:
        tabela.to_excel(writer,index=False,sheet_name='Clientes')
    st.download_button('📥 Baixar lista de clientes em Excel',buffer.getvalue(),file_name='clientes_por_situacao.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    with st.expander('🔎 Conferência dos números (área técnica)'):
        st.write(f'Grupos com última compra desde 01/01/2024: **{_num(len(dados))}**')
        st.write(f'Grupos com histórico suficiente para classificação: **{_num(total)}**')
        st.write(f'Grupos fora da base de classificação: **{_num(len(dados)-total)}**')
        st.write(f'Grupos que voltaram a comprar na coleção atual: **{_num(len(recuperados))}**')
        st.write('A base de classificação exclui grupos novos que não existiam antes da última coleção encerrada. Não confunda essa diferença com clientes sem compra desde 2024.')
