# %%
import gurobipy as gp
import pandas as pd
import numpy as npy
import os
import time
import sys

t_out = 1.5*60*60
#meses = ['Jan', 'Fev', 'Mar' , 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'] 

################################################
################################################
################################################
################################################
######### HEURISTICA RELAX AND FIX #############
################################################
################################################
################################################

# %%
dir = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Dados Gerados Mestrado\Auxiliar'

df_cluster_cds_cities = pd.read_csv(os.path.join(dir, 'df_cluster_cds_cities.csv'))
df_cluster_pa_cities = pd.read_csv(os.path.join(dir, 'df_cluster_pa_cities.csv'))

demanda_df = pd.read_csv(os.path.join(dir, 'df_instalacao.csv'))
demanda_reversa_df = pd.read_csv(os.path.join(dir, 'df_desinstalacao.csv'))
distancia_cd_pa_df = pd.read_csv(os.path.join(dir, 'df_dist_CD_PA.csv'))
distancia_cd_desc_df = pd.read_csv(os.path.join(dir, 'df_dist_CD_Desc.csv'))
distancia_fab_cd_df = pd.read_csv(os.path.join(dir, 'df_dist_Fab_CD.csv'))
distancia_pa_zd_df = pd.read_csv(os.path.join(dir, 'df_dist_ZD_PA.csv'))
df_regiao_pa_zd = pd.read_csv(os.path.join(dir, 'df_regiao_pa_zd.csv')) # OK
df_estq_inicial = pd.read_csv(os.path.join(dir, 'df_estq_inicial.csv'))

def regiao_cd_pa():
    df_regiao_cd_pa = pd.DataFrame(npy.ones(distancia_cd_pa_df.loc[:, 'CD':].shape).astype(int), columns=distancia_cd_pa_df.loc[:, 'CD':].columns)
    df_regiao_cd_pa['CD'] = distancia_cd_pa_df['CD']
    return df_regiao_cd_pa
df_regiao_cd_pa = regiao_cd_pa()

# %%
def select_cols(df):
    for each in df.columns:
        if each in ['CD', 'PA', 'ZD', 'FAB', 'DESC']:
            df = df.loc[:, each:]
            break
    return df

# Lista com todos os DataFrames
dfs = [
    demanda_df, demanda_reversa_df, distancia_cd_pa_df, df_regiao_cd_pa,
    distancia_cd_desc_df, distancia_fab_cd_df, distancia_pa_zd_df
]

# Aplicar a função select_cols a cada DataFrame na lista
dfs = [select_cols(df) for df in dfs]

# Descompactar a lista de volta em variáveis separadas
(   demanda_df, demanda_reversa_df, distancia_cd_pa_df, df_regiao_cd_pa,
    distancia_cd_desc_df, distancia_fab_cd_df, distancia_pa_zd_df
) = dfs

# %%
try:
    demanda_df = demanda_df.drop(columns=['demanda_anual'])
    demanda_reversa_df = demanda_reversa_df.drop(columns=['demanda_anual'])
except KeyError:
    print('Coluna inexistente')# Se a coluna não existir, apenas continue
demanda_df.columns = ['ZD'] + [f'Dem_T{i:02}' for i in range(1, len(demanda_df.columns))]
demanda_reversa_df.columns = ['ZD'] + [f'Dem_T{i:02}' for i in range(1, len(demanda_df.columns))]

# %%
# Fabricantes
num_fabricante = distancia_fab_cd_df['FAB'].count()
rng_estq_fab = [10000, 10002]
mrf = 9000
maf = 9000
ef = npy.ceil(rng_estq_fab[1]*2.4)  # estf substituído por ef
rf = round(0.2, 1)

# CDs
num_cds_possiveis = df_regiao_cd_pa['CD'].count()
num_cds_sel = 1
rng_estq_cd = [149999, 150001]
rng_rest_cd = [rng_estq_cd[0]*0.75, rng_estq_cd[1]*0.75]

ec = npy.ceil(rng_estq_cd[1]*2.1)  # estc substituído por ec
min_estq_bom_cd = npy.ceil(ec*0.2)
nc = num_cds_sel  # mc substituído por nc
rc = round(0.7, 1)

# PAs
num_postos_possiveis = sum(1 for each in df_regiao_cd_pa.columns if 'PA_' in each)
num_postos_sel = int(num_postos_possiveis * 0.7)
np = int(num_postos_possiveis * 0.7)  # mp substituído por np
# rng_estq_pa = [6000, 7000]
# rng_rest_pa = [4000, 5000]
ep = {k:v for k, v in zip(df_estq_inicial['PA'], df_estq_inicial['lim_estq_in_pa'])}  # estp substituído por ep
# min_estq_bom_pa = {k:npy.ceil((v*0.02)) for k, v in zip(df_estq_inicial['PA'], df_estq_inicial['lim_estq_in_pa'])}

# Descartes
num_desc_total = distancia_cd_desc_df['DESC'].count()
dr = 1000000
rd = round(1 - rc - rf, 1)

# Clientes
clientes = distancia_pa_zd_df['ZD'].count()

# Tempo
periodo = sum(1 for each in demanda_df.columns if 'Dem_' in each)

# %%
# Fabricantes
print('# Fabricantes')
print(f'Número de fabricantes: {num_fabricante}')
print(f'Faixa de estoque do fabricante: {rng_estq_fab}')
print(f'Restrição de fabricante: {mrf}')
print(f'Faixa de estoque fixo: {maf}')
print(f'Estoque final: {ef}')  # estf substituído por ef
print(f'Fator de risco do fabricante: {rf}')

# CDs
print('# CDs')
print(f'Número de CDs possíveis: {num_cds_possiveis}')
print(f'Número de CDs selecionados: {num_cds_sel}')
print(f'Estoque do CD: {rng_estq_cd}')
print(f'Restrição de estoque do CD: {rng_rest_cd}')
print(f'Estoque do CD: {ec}')  # estc substituído por ec
print(f'Máximo de CD: {nc}')  # mc substituído por nc
print(f'Fator de risco do CD: {rc}')

# PAs
print('# PAs')
print(f'Número de postos possíveis: {num_postos_possiveis}')
print(f'Número de postos selecionados: {num_postos_sel}')
# print(f'Faixa de estoque do PA: {rng_estq_pa}')
# print(f'Restrição de estoque do PA: {rng_rest_pa}')

# Descartes
print('# Descartes')
print(f'Número total de descartes: {num_desc_total}')
print(f'Restrição de descartes: {dr}')
print(f'Fator de risco do descarte: {rd}')

# Clientes
print('# Clientes')
print(f'Número de clientes: {clientes}')
# print(f'Estoque do cliente: {ep}')  # estp substituído por ep
print(f'Máximo de clientes: {np}')  # mp substituído por np

# Tempo
print('# Tempo')
print(f'Período: {periodo}')

# Custos
print('# Custos')
# print(f'Custo de manutenção do CD: {vi}')  # c_man_cd substituído por vi
# print(f'Custo de abertura do CD: {hi}')  # c_ab_cd substituído por hi
# print(f'Custo de manutenção do PA: {xp}')  # c_man_pa substituído por xp
# print(f'Custo de abertura do PA: {up}')  # c_ab_pa substituído por up
# print(f'Custo k: {k}')

# %%
def criar_base_aleatoria(nome_linhas, nome_colunas_def, nome_colunas, list_range):
    qt_linhas = len(nome_linhas)
    qt_colunas = len(nome_colunas)
    df = pd.DataFrame()
    for i in range(len(nome_colunas_def)):
        df[nome_colunas_def[i]] = [nome_linhas[i] for i in range(len(nome_linhas))]
    dados_ger = npy.random.randint(list_range[0], list_range[1], size=(qt_linhas, qt_colunas))
    for j in range(qt_colunas):
        df[f'{nome_colunas[j]}'] = dados_ger[:, j]
    return df

nome_colunas_def = ['ZD']

######### Criação de restrições de PA #########
restricoes_pa_df = df_estq_inicial.loc[:,['PA', 'Restr_Rec', 'Restr_Env']]
restricoes_pa_df = restricoes_pa_df.rename(columns={'PA': 'Postos_Avançados'})
# list_range = rng_rest_pa
# postos = [f'PA_{i+1}' for i in range(num_postos_possiveis)]
# nome_colunas = ['Restr_Rec', 'Restr_Env']
# restricoes_pa_df = criar_base_aleatoria(postos, nome_colunas_def, nome_colunas, list_range)
# del list_range

######### Criação de restrições de CD #########
list_range = rng_rest_cd
nome_colunas = ['Restr_Rec', 'Restr_Env']
distribuidores = [f'Distribuidor_{i+1}' for i in range(num_cds_possiveis)]
restricoes_cd_df = criar_base_aleatoria(distribuidores, nome_colunas_def, nome_colunas, list_range)

del list_range

######### Criação de estoque inicial entre CD #########
list_range = rng_estq_cd # Custos aleatórios entre 10 e 50
nome_colunas_def = ['Distribuidores']
nome_colunas = ['Est_Ini_g', 'Est_Ini_f', 'Est_ini_d']
distribuidores = [f'Distribuidor_{i+1}' for i in range(num_cds_possiveis)]
estoques_cd_df = criar_base_aleatoria(distribuidores, nome_colunas_def, nome_colunas, list_range)
estoques_cd_df['Est_Ini_g'] = estoques_cd_df['Est_Ini_g']*0.7
estoques_cd_df['Est_Ini_f'] = estoques_cd_df['Est_Ini_f']*0.2
estoques_cd_df['Est_ini_d'] = estoques_cd_df['Est_ini_d']*0.1

######### Criação de estoque inicial entre CD #########
list_range = rng_estq_fab # Custos aleatórios entre 10 e 50
nome_colunas_def = ['Distribuidores']
nome_colunas = ['Est_Ini_x_fab', 'Est_Ini_c_fab']
fabricantes = [f'F_{i+1}' for i in range(num_fabricante)]
estoques_fab_df = criar_base_aleatoria(fabricantes, nome_colunas_def, nome_colunas, list_range)

######### Criação de estoque inicial PA #########
estoques_pa_df = df_estq_inicial.loc[:,['PA', 'estq_in_b_pa', 'estq_in_r_pa']]
estoques_pa_df = estoques_pa_df.rename(columns={'PA': 'Postos_Avançados', 'estq_in_b_pa': 'Est_Ini_c_PA', 'estq_in_r_pa': 'Est_Ini_x_PA'})

# %%
# Rótulos dos Índices
CDs = list()
for i in range(num_cds_possiveis):
    CDs.append("CD_{:04}".format(i + 1))

# Rótulos dos PAs
PAs = list()
for j in range(num_postos_possiveis):
    PAs.append("PA_{:04}".format(j + 1))

# Rótulos dos Descartes
Ds = list()
for j in range(num_desc_total):
    Ds.append("D_{:04}".format(j + 1))

# Rótulos dos Fabricantes
Fs = list()
for j in range(num_fabricante):
    Fs.append("F_{:04}".format(j + 1))

# Rótulos dos PAs
ZD = list()
for j in range(clientes):
    ZD.append("ZD_{:04}".format(j + 1))

# Rótulos dos Periodos
periodo_list = list()
for t in range(0, periodo):
    num_periodo = t #str(t+1).zfill(2)
    periodo_list.append(num_periodo)

# %%
# Demanda avanço produtos próprios
demrj = dict()
demrj = {ZD[j]: {t: demanda for t, demanda in zip(periodo_list, demanda_reversa_df.iloc[j,1:])} for j in range(clientes)}

# Demanda reversa produtos impróprios
demaj = dict()
demaj = {ZD[j]: {t: demanda for t, demanda in zip(periodo_list, demanda_df.iloc[j,1:])} for j in range(clientes)}

# Dicionario de restrições de avanço pa
map = dict()
map = {pa: restricoes_pa_df.loc[i, 'Restr_Env'] for i, pa in enumerate(PAs[:num_postos_possiveis])}

# Dicionario de restrições de reversa pa
mrp = dict()
mrp = {pa: restricoes_pa_df.loc[i, 'Restr_Rec'] for i, pa in enumerate(PAs[:num_postos_possiveis])}

# Dicionario de restrições de avanço
mac = dict()
mac = {CDs[i]:restricoes_cd_df.iloc[i,2] for i in range(num_cds_possiveis)}

# Dicionario de restrições de reversa
mrc = dict()
mrc = {CDs[i]:restricoes_cd_df.iloc[i,1] for i in range(num_cds_possiveis)}

# Dicionario de distâncias
dcp = dict()
dcp = {CDs[j]: {pa:dist for pa, dist in zip(PAs, distancia_cd_pa_df.iloc[j,1:])} for j in range(num_cds_possiveis)}

# Dicionario de indicação mesma região
ccp = dict()
ccp = {CDs[j]: {pa:indbin for pa, indbin in zip(PAs, df_regiao_cd_pa.iloc[j,1:])} for j in range(num_cds_possiveis)}

# Dicionario de estoque inicial CDs g
estc_g_in = dict()
estc_g_in = {cd:each_est for cd, each_est in zip(CDs, estoques_cd_df.iloc[:,1])}

# Dicionario de estoque inicial CDs f
estc_f_in = dict()
estc_f_in = {cd:each_est for cd, each_est in zip(CDs, estoques_cd_df.iloc[:,2])}

# Dicionario de estoque inicial CDs d
estc_d_in = dict()
estc_d_in = {cd:each_est for cd, each_est in zip(CDs, estoques_cd_df.iloc[:,3])}

# Dicionario de distancia entre CDs e descartes
dcd = dict()
dcd = {CDs[j]: {d:indbin for d, indbin in zip(Ds, distancia_cd_desc_df.iloc[:,j+1])} for j in range(num_cds_possiveis)}

# Dicionario de estoque inicial fab x
estf_in_x = dict()
estf_in_x = {f:each_est for f, each_est in zip(Fs, estoques_fab_df.iloc[:,1])}

# Dicionario de estoque inicial fab c
estf_in_c = dict()
estf_in_c = {f:each_est for f, each_est in zip(Fs, estoques_fab_df.iloc[:,2])}

# Dicionario de distancia entre fab e CDs
dfc = dict()
dfc = {Fs[j]: {d:indbin for d, indbin in zip(CDs, distancia_fab_cd_df.iloc[j,1:])} for j in range(num_fabricante)}

# Dicionario de distancia entre PA e ZD
dpj = dict()
dpj = {PAs[j]: {d:indbin for d, indbin in zip(ZD, distancia_pa_zd_df.iloc[:,j+1])} for j in range(num_postos_possiveis)}

# Dicionario de estoque inicial PA
estp_x_in = dict()
estp_x_in = {pa:each_est for pa, each_est in zip(PAs, estoques_pa_df.iloc[:,1])}

# Dicionario de estoque inicial PAa
estp_g_in = dict()
estp_g_in = {pa:each_est for pa, each_est in zip(PAs, estoques_pa_df.iloc[:,2])}

# Dicionario de indicação mesma região
cpj = dict()
cpj = {PAs[j]: {z:indbin for z, indbin in zip(ZD, df_regiao_pa_zd.iloc[j,1:])} for j in range(num_postos_possiveis)}

# %%
# Criar o modelo em branco
m = gp.Model()
m.setParam('NodefileStart', 5)  # Começa a usar disco após 4GB de RAM
m.setParam('NodefileDir', rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\arquivos temporarios')  # Onde salvar
m.Params.MIPGap = 0.05  # Gap de 5%
m.Params.Threads = 4
m.Params.LogToConsole = 1
m.Params.Presolve = 2
m.Params.Method = 1

# Periodo
T = range(0, periodo)
T_menos_1 = range(1, periodo)

BMCD = ec

BM = []
for t in T:
    remaining_demand = 0
    for tau in range(t, periodo):  # Soma das demandas do período t em diante
        for j in ZD:
            remaining_demand += demaj[j][t]
    BM.append(remaining_demand)

lote_fab_desc = 500

######### Adicionar as variáveis de decisão CD
oc = m.addVars(CDs, periodo_list, vtype=gp.GRB.BINARY)  # Abertura do CD

bcp = m.addVars(CDs, PAs, periodo_list, vtype=gp.GRB.BINARY)  # variável para garantir o atendimento do PA, por pelo menos 1 CD
bcd = m.addVars(CDs, Ds, periodo_list, vtype=gp.GRB.BINARY)  # variável para garantir o atendimento do PA, por pelo menos 1 CD

z = m.addVars(CDs, periodo_list, vtype=gp.GRB.BINARY)  # Primeira abertura CD

qap = m.addVars(CDs, PAs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qcp substituído por qap
qrd = m.addVars(CDs, Ds, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qcd substituído por qrd
qrf = m.addVars(CDs, Fs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qcf substituído por qrf

sc = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestc substituído por sc
scrf = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestcf substituído por scrf
scrd = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestcd substituído por scrd
scbp = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestcg substituído por scbp
scbt = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestcg substituído por scbp

# # First, add auxiliary integer variables
# l_atv_fab = m.addVars(CDs, Fs, periodo_list, vtype=gp.GRB.BINARY)  # variável para garantir o atendimento do CD, por pelo menos 1 fabricante
# #q_lote_fab = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qlote substituído por q_lote_fab
# q_lote_fab = m.addVars(CDs, Fs, periodo_list, lb=0, vtype=gp.GRB.INTEGER)

# l_atv_desc = m.addVars(CDs, periodo_list, vtype=gp.GRB.BINARY)  # variável para garantir o atendimento do CD, por pelo menos 1 fabricante
# # q_lote_desc = m.addVars(CDs, periodo_list, lb=0, vtype=gp.GRB.INTEGER)  # qlote substituído por q_lote_fab
# q_lote_desc = m.addVars(CDs, periodo_list, vtype=gp.GRB.INTEGER)

######### Adicionar as variáveis de decisão PA
op = m.addVars(PAs, periodo_list, vtype=gp.GRB.BINARY)  # Abertura do PA
bpj = m.addVars(PAs, ZD, periodo_list, vtype=gp.GRB.BINARY)  # variável para garantir o atendimento do ZD, por pelo menos 1 PA

qrc = m.addVars(PAs, CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qpc substituído por qrc
qaj = m.addVars(PAs, ZD, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qpj substituído por qaj
qrp = m.addVars(ZD, PAs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qjp substituído por qrp

sp = m.addVars(PAs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestp substituído por sp
sprc = m.addVars(PAs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestpx substituído por sprc
spbj = m.addVars(PAs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestpg substituído por spbj

w = m.addVars(PAs, periodo_list, vtype=gp.GRB.BINARY)
# msrc = m.addVars(PAs, vtype=gp.GRB.INTEGER)
# msbj = m.addVars(PAs, vtype=gp.GRB.INTEGER)
# qmin_dem_atendida_pa substituído por min_dem_atendida_pa

######### Adicionar as variáveis de decisão Fabricantes
qac = m.addVars(Fs, CDs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qfc substituído por qac
qfra = m.addVars(Fs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qfxg substituído por qfra
sf = m.addVars(Fs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestf substituído por sf
sfrf = m.addVars(Fs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestfx substituído por sfrf
sfbc = m.addVars(Fs, periodo_list, lb=0, vtype=gp.GRB.CONTINUOUS)  # qestfc substituído por sfbc

######### Custos #########
vi = 100000  # c_man_cd substituído por vi
hi = 3000000  # c_ab_cd substituído por hi
xp = {k:v for k,v in zip(df_estq_inicial['PA'], df_estq_inicial['custo_man_por_faixa'])}  # c_man_pa substituído por xp
up = {k:v for k,v in zip(df_estq_inicial['PA'], df_estq_inicial['custo_por_faixa'])}  # c_ab_pa substituído por up
k = 1.5

# %%
######### Definir a função objetivo
m.setObjective(
    gp.quicksum(
        gp.quicksum(hi * z[i, t] for i in CDs) +  # c_ab_cd substituído por hi
        gp.quicksum(vi * oc[i, t] for i in CDs) +  # c_man_cd substituído por vi
        gp.quicksum(up[p] * w[p, t] for p in PAs) +  # c_ab_pa substituído por up
        gp.quicksum(xp[p] * op[p, t] for p in PAs) +  # c_man_pa substituído por xp
        gp.quicksum(qac[l, i, t] * dfc[l][i] * k for l in Fs for i in CDs) +  # qfc substituído por qac
        gp.quicksum(qrf[i, l, t] * dfc[l][i] * k for l in Fs for i in CDs) +  # qcf substituído por qrf
        gp.quicksum(qap[i, p, t] * dcp[i][p] * k for i in CDs for p in PAs) +  # qcp substituído por qap
        gp.quicksum(qrc[p, i, t] * dcp[i][p] * k for p in PAs for i in CDs) +  # qpc substituído por qrc
        gp.quicksum(qrd[i, d, t] * dcd[i][d] * k for i in CDs for d in Ds) +  # qcd substituído por qrd
        gp.quicksum(qaj[p, j, t] * dpj[p][j] * k for p in PAs for j in ZD) +  # qpj substituído por qaj
        gp.quicksum(qrp[j, p, t] * dpj[p][j] * k for p in PAs for j in ZD)  # qjp substituído por qrp
        for t in T
    ),
    sense=gp.GRB.MINIMIZE
)

# %%
############### Fabricantes ###############

# Capacidade máxima de recebimento Fab 3.2
f010 = m.addConstrs(gp.quicksum(qrf[i, l, t] for i in CDs) <= mrf for l in Fs for t in T)  # qcf substituído por qrf

# Capacidade máxima de envio Fab 3.3
f020 = m.addConstrs(gp.quicksum(qac[l, i, t] for i in CDs) <= maf for l in Fs for t in T)  # qfc substituído por qac

# Composição de estoque Fab 3.4
f030 = m.addConstrs(sfbc[l, t] + sfrf[l, t] == sf[l, t] for l in Fs for t in T)  # qestfc, qestfx, qestf substituídos por sfbc, sfrf, sf

# Balanceamento de estoque ruim 3.5
f041 = m.addConstrs(sfrf[l, 0] == estf_in_x[l] for l in Fs)  # qestfx substituído por sfrf
f042 = m.addConstrs(gp.quicksum(qrf[i, l, t] for i in CDs) - qfra[l, t] + sfrf[l, t - 1] == \
                    sfrf[l, t] for l in Fs for t in T_menos_1)  # qcf substituído por qrf, qfxg substituído por qfra, qestfx substituído por sfrf

# Balanceamento de estoque bom 3.6
f051 = m.addConstrs(sfbc[l, 0] == estf_in_c[l] for l in Fs)  # qestfc substituído por sfbc
f052 = m.addConstrs(qfra[l, t] - gp.quicksum(qac[l, i, t] for i in CDs) + sfbc[l, t - 1] == \
                    sfbc[l, t] for l in Fs for t in T_menos_1)  # qfxg substituído por qfra, qfc substituído por qac, qestfc substituído por sfbc

# Estoque não seja maior que a capacidade 3.7
f060 = m.addConstrs(sf[l, t] <= ef for l in Fs for t in T)  # qestf substituído por sf, estf substituído por ef

# Big M 3.8
f070 = m.addConstrs(gp.quicksum(qac[l, i, t] for l in Fs) <= sum(BM[t] for t in T) * oc[i, t] for i in CDs for t in T)  
# qfc substituído por qac, M substituído por BM

# %%
############### CDs ###############

# Capacidade máxima de recebimento CD 3.9
c010 = m.addConstrs(gp.quicksum(qac[l, i, t] for l in Fs) + gp.quicksum(qrc[p, i, t] for p in PAs) <= mrc[i] for i in CDs for t in T)  # qfc substituído por qac, qpc substituído por qrc

# Capacidade máxima de envio CD 3.10
c020 = m.addConstrs(gp.quicksum(qrf[i, l, t] for l in Fs) + gp.quicksum(qap[i, p, t] for p in PAs) + 
                    gp.quicksum(qrd[i, d, t] for d in Ds) <= mac[i] for i in CDs for t in T)  # qcf substituído por qrf, qcp substituído por qap, qcd substituído por qrd

# Composição de estoque CD 3.11
c030 = m.addConstrs(scrd[i, t] + scrf[i, t] + scbp[i, t] + scbt[i, t] == sc[i, t] for i in CDs for t in T)
c031 = m.addConstrs(scrd[i, t] <= scrf[i, t] for i in CDs for t in T)
c033 = m.addConstrs(scrf[i, t] + scrd[i, t] <= scbp[i, t]*0.65 for i in CDs for t in T)


# Balanceamento de estoque CD 
# 3.12
c041 = m.addConstrs(scrf[i, 0] == estc_f_in[i] for i in CDs)  # qestcf substituído por scrf
c042 = m.addConstrs((gp.quicksum(qrc[p, i, t] for p in PAs) * rf) - gp.quicksum(qrf[i, l, t] for l in Fs) 
                         + scrf[i, t - 1] == scrf[i, t] for i in CDs for t in T_menos_1)  # qpc substituído por qrc, qcf substituído por qrf, qestcf substituído por scrf

# # #----------- Novo -----------
# # 1. Quantidade enviada deve ser múltipla do lote
# c043 = m.addConstrs(qrf[i, l, t] == lote_fab_desc * q_lote_fab[i, l, t] for i in CDs for l in Fs for t in T)
# # 3. Se tivermos estoque (l_atv_fab=1), garantimos que ele seja pelo menos 1 lote
# c044 = m.addConstrs(scrf[i,t] >= lote_fab_desc * l_atv_fab[i,l,t] for i in CDs for l in Fs for t in T)
# # 4. Calcula quantos lotes completos temos no estoque (arredondando para baixo)
# # Usamos a função floor através de uma variável inteira auxiliar
# c045 = m.addConstrs(q_lote_fab[i,l,t] <= scrf[i,t]/lote_fab_desc for i in CDs for l in Fs for t in T)
# #c046 = m.addConstrs(q_lote_fab[i,l,t] >= scrf[i,t]/lote_fab_desc - 1 + 1e-6 for i in CDs for l in Fs for t in T)
# # 5. O número de lotes enviados não pode exceder os lotes disponíveis no estoque
# c047 = m.addConstrs(q_lote_fab[i,l,t] <= q_lote_fab[i,l,t] for i in CDs for l in Fs for t in T)
# # 6. Forçar envio de lotes completos quando possível
# # Se o estoque for maior que um lote, devemos enviar pelo menos um lote
# c048 = m.addConstrs(q_lote_fab[i,l,t] >= l_atv_fab[i,l,t] for i in CDs for l in Fs for t in T)
# c049 = m.addConstrs(scrf[i,t] <= (lote_fab_desc - 1) + BMCD * l_atv_fab[i,l,t] for i in CDs for l in Fs for t in T)

# 3.13
c051 = m.addConstrs(scrd[i, 0] == estc_d_in[i] for i in CDs)  # qestcd substituído por scrd
c052 = m.addConstrs(gp.quicksum(qrc[p, i, t] for p in PAs) * rd - gp.quicksum(qrd[i, d, t] for d in Ds)
                        + scrd[i, t - 1] == scrd[i, t] for i in CDs for t in T_menos_1)  # qpc substituído por qrc, qcd substituído por qrd, qestcd substituído por scrd

# # #----------- Novo -----------
# # 1. Quantidade enviada deve ser múltipla do lote
# c053 = m.addConstrs(qrd[i, d, t] == lote_fab_desc * q_lote_desc[i, t] for i in CDs for d in Ds for t in T)
# # 3. Se tivermos estoque (l_atv_fab=1), garantimos que ele seja pelo menos 1 lote
# c054 = m.addConstrs(scrd[i,t] >= lote_fab_desc * l_atv_desc[i,t] for i in CDs for t in T)
# # 4. Calcula quantos lotes completos temos no estoque (arredondando para baixo)
# # Usamos a função floor através de uma variável inteira auxiliar
# c055 = m.addConstrs(q_lote_desc[i,t] <= scrd[i,t]/lote_fab_desc for i in CDs for t in T)
# # c056 = m.addConstrs(q_lote_desc[i,t] >= scrd[i,t]/lote_fab_desc - 1 + 1e-6 for i in CDs for t in T)
# # 5. O número de lotes enviados não pode exceder os lotes disponíveis no estoque
# c057 = m.addConstrs(q_lote_desc[i,t] <= q_lote_desc[i,t] for i in CDs for t in T)
# # 6. Forçar envio de lotes completos quando possível
# # Se o estoque for maior que um lote, devemos enviar pelo menos um lote
# c058 = m.addConstrs(q_lote_desc[i,t] >= l_atv_desc[i,t] for i in CDs for t in T)
# c059 = m.addConstrs(scrd[i,t] <= (lote_fab_desc - 1) + BMCD * l_atv_desc[i,t] for i in CDs for t in T)

# 3.14
c061 = m.addConstrs(scbp[i, 0] == estc_g_in[i] for i in CDs)  # qestcg substituído por scbp
c062 = m.addConstrs(scbt[i, t] == gp.quicksum(qrc[p, i, t] for p in PAs) * rc for i in CDs for t in T)  # qestcg substituído por scbp
c063 = m.addConstrs(scbp[i, t-1] - gp.quicksum(qap[i, p, t] for p in PAs) + scbt[i, t-1] == scbp[i, t]
                    for i in CDs for t in T_menos_1)
c064 = m.addConstrs(scbp[i, t] >= min_estq_bom_cd*0.5 for i in CDs for t in T)  # qestcg substituído por scbp

# Estoque menor que a capacidade do CD 3.15
c070 = m.addConstrs(sc[i, t] <= ec for i in CDs for t in T)  # qestc substituído por sc, estc substituído por ec

# Garantir que o PA seja atendido por pelo menos 1 CD 3.16
c080 = m.addConstrs(gp.quicksum(bcp[i, p, t] for i in CDs) == 1 for p in PAs for t in T)

# Manutenção da Abertura do CD 3.17
c090 = m.addConstrs(oc[i, t] >= oc[i, t - 1] for i in CDs for t in T_menos_1)

# Big M
# 3.18
c100 = m.addConstrs(qrd[i, d, t] <= BMCD * oc[i, t] for i in CDs for d in Ds for t in T)  # qcd substituído por qrd, M substituído por BM
# 3.19
c110 = m.addConstrs(qap[i, p, t] <= BMCD * oc[i, t] for i in CDs for p in PAs for t in T)  # qcp substituído por qap, M substituído por BM
# 3.20
c120 = m.addConstrs(qrf[i, l, t] <= BMCD * oc[i, t] for i in CDs for l in Fs for t in T)  # qcf substituído por qrf, M substituído por BM
# 3.21
c130 = m.addConstrs(qrd[i, d, t] <= BMCD * bcd[i, d, t] for i in CDs for d in Ds for t in T)  # qcd substituído por qrd, M substituído por BM
# 3.22
c140 = m.addConstrs(qap[i, p, t] <= BMCD * bcp[i, p, t] for i in CDs for p in PAs for t in T)  # qcp substituído por qap, M substituído por BM

# Garantir Primeira Abertura do CD 3.23
c151 = m.addConstrs(z[i, 0] == oc[i, 0] for i in CDs) # oc[i, 0]
c152 = m.addConstrs(z[i, t] >= oc[i, t] - oc[i, t - 1] for i in CDs for t in T_menos_1)

# Limitar a quantidade de abertura de CD 3.24
c160 = m.addConstr(gp.quicksum(z[i, t] for i in CDs for t in T) <= nc)  # mc substituído por nc

# Garantir que ocorra abertura e atendimento apenas na mesma região 3.25
c170 = m.addConstrs(bcp[i, p, t] <= ccp[i][p] * oc[i, t] for i in CDs for p in PAs for t in T)

# Restrição: Garantir que cada PA seja atendido pelo mesmo CD durante todo o período 3.26
c180 = m.addConstrs(
    bcp[i, p, t] >= bcp[i, p, t-1]  # O CD que atende o PA no período 0 deve continuar atendendo nos períodos subsequentes
    for i in CDs for p in PAs for t in T if t > 0
)

# %%
# min_dem_atendida_pa = 50
# Variáveis

############### PAs ###############
# Garantir não estourar o recebimento do CD 3.27
p010 = m.addConstrs(gp.quicksum(qrp[j, p, t] for j in ZD) + gp.quicksum(qap[i, p, t] for i in CDs) <= mrp[p] for p in PAs for t in T)  # qjp substituído por qrp, qcp substituído por qap

# Maximo envio do PA 3.28
p020 = m.addConstrs(gp.quicksum(qaj[p, j, t] for j in ZD) + gp.quicksum(qrc[p, i, t] for i in CDs) <= map[p] for p in PAs for t in T)  # qpj substituído por qaj, qpc substituído por qrc

# Estoque 3.29
p030 = m.addConstrs(spbj[p, t] + sprc[p, t] == sp[p, t] for p in PAs for t in T)  # qestpg substituído por spbj, qestpx substituído por sprc, qestp substituído por sp
p031 = m.addConstrs(spbj[p, t] >= spbj[p, 0] * 0.1 for p in PAs for t in T)

# Balanceamento de estoque impróprio 3.30
p041 = m.addConstrs(sprc[p, 0] == estp_x_in[p] for p in PAs)  # qestpx substituído por sprc
p042 = m.addConstrs(gp.quicksum(qrp[j, p, t] for j in ZD) - gp.quicksum(qrc[p, i, t] for i in CDs) + sprc[p, t - 1] == 
                    sprc[p, t] for p in PAs for t in T_menos_1)  # qjp substituído por qrp, qpc substituído por qrc, qestpx substituído por sprc

# Balanceamento de estoque bom 3.31
p051 = m.addConstrs(spbj[p, 0] == estp_g_in[p] for p in PAs)  # qestpg substituído por spbj
p052 = m.addConstrs(gp.quicksum(qap[i, p, t] for i in CDs) - gp.quicksum(qaj[p, j, t] for j in ZD) + spbj[p, t - 1] == 
                    spbj[p, t] for p in PAs for t in T_menos_1)  # qcp substituído por qap, qpj substituído por qaj, qestpg substituído por spbj, qestpx substituído por sprc
# p053 = m.addConstrs(spbj[p, t] >= min_estq_bom_pa[p] for p in PAs for t in T)  # qestpg substituído por spbj

# Garantir o não estouro do estoque 3.32
p060 = m.addConstrs(sp[p, t] <= ep[p] for p in PAs for t in T)  # qestp substituído por sp, estp substituído por ep

# Garantir atendimento total da demanda 3.33
p070 = m.addConstrs(gp.quicksum(qrp[j, p, t] for p in PAs) == demrj[j][t] for j in ZD for t in T)  # qjp substituído por qrp
# p071 = m.addConstrs(gp.quicksum(qrp[j, p, t] for j in ZD) >= min_dem_atendida_pa * op[p, t] for p in PAs for t in T)  # qjp substituído por qrp

# Garantir atendimento total da demanda 3.34
p080 = m.addConstrs(gp.quicksum(qaj[p, j, t] for p in PAs) == demaj[j][t] for j in ZD for t in T)  # qpj substituído por qaj

# Abertura do PA 3.35
p090 = m.addConstrs(op[p, t] >= op[p, t - 1] for p in PAs for t in T_menos_1)

# Abertura nas mesmas regiões 3.36
p100 = m.addConstrs(bpj[p, j, t] <= cpj[p][j] * op[p, t] for p in PAs for j in ZD for t in T)

# Garantir a primeira abertura 3.37
p111 = m.addConstrs(w[p, 0] == op[p, 0] for p in PAs)
p112 = m.addConstrs(w[p, t] >= op[p, t] - op[p, t - 1] for p in PAs for t in T_menos_1)

# Garantir que a abertura do PA não exceda o máximo permitido 3.38
p120 = m.addConstr(gp.quicksum(w[p, t] for p in PAs for t in T) <= np)  # mp substituído por np

# Garantir o atendimento de uma zona de demanda por um e apenas um PA 3.39
p130 = m.addConstrs(gp.quicksum(bpj[p, j, t] for p in PAs) == 1 for j in ZD for t in T)

# Restr Big M
p140 = m.addConstrs(qrp[j, p, t] <= mrp[p] * op[p, t] for j in ZD for p in PAs for t in T)  # qjp substituído por qrp, M substituído por BM
p180 = m.addConstrs(qaj[p, j, t] <= map[p] * bpj[p, j, t] for p in PAs for j in ZD for t in T)  # qpj substituído por qaj, M substituído por BM
p150 = m.addConstrs(qaj[p, j, t] <= map[p] * op[p, t] for p in PAs for j in ZD for t in T)  # qpj substituído por qaj, M substituído por BM
p160 = m.addConstrs(qrc[p, i, t] <= map[p] * op[p, t] for p in PAs for i in CDs for t in T)  # qpc substituído por qrc, M substituído por BM
p170 = m.addConstrs(qrp[j, p, t] <= mrp[p] * bpj[p, j, t] for j in ZD for p in PAs for t in T)  # qjp substituído por qrp, M substituído por BM
p190 = m.addConstrs(qrc[p, i, t] <= map[p] * bcp[i, p, t] for p in PAs for i in CDs for t in T)  # qpc substituído por qrc, M substituído por BM

# Garantir que uma ZD seja atendida pela mesma PA durante todo o período 3.46
p200 = m.addConstrs(
    bpj[p, j, t] >= bpj[p, j, t-1] 
    for p in PAs for j in ZD for t in T if t > 0
)

# # Garantir que um PA não atenda mais de 30 ZD
# p210 = m.addConstrs(
#     gp.quicksum(bpj[p, j, t] for j in ZD) <= 15
#     for p in PAs for t in T
# )

# %%
############### Descartes ###############

# Maximo recebimento no descarte
d001 = m.addConstrs(gp.quicksum(qrd[i, d, t] for i in CDs) <= dr for d in Ds for t in T)  # qcd substituído por qrd

# %%
class MultiOutput:
    def __init__(self, *outputs):
        self.outputs = outputs

    def write(self, text):
        for output in self.outputs:
            output.write(text)

    def flush(self):
        for output in self.outputs:
            if hasattr(output, 'flush'):
                output.flush()

# %%
m.setParam('TimeLimit', t_out)

# Abrir o arquivo de log
log_file = open(rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\log_optmize.txt', 'w')

# Salvar a saída padrão original
original_stdout = sys.stdout

# Redirecionar a saída para o console e para o arquivo
sys.stdout = MultiOutput(sys.stdout, log_file)

# Marcação de tempo antes da otimização
start_time = time.time()
print(start_time)

m.optimize()

end_time = time.time()

# Restaurar a saída padrão para o console
sys.stdout = original_stdout

# Fechar o arquivo de log
log_file.close()

# Calculando o tempo total
total_time = end_time - start_time
print(f"Tempo total de execução: {total_time:.2f} segundos")

# for k in range(K):
#   if y[k].X > 0:
#     #print("\nUsou a barra:\t", k, y[k].X, end = '')
#     print("\nUsou a barra:\t", str(k+1), end = '')

# print("barra\titem\tquantidade")
# for k in range(K):
#   for j in range(N):
#      if a[j,k].X > 0:
#       print(str(k+1)+"\t"+str(j+1)+"\t"+str(round(a[j,k].X)))

# %%
try:
    lower_bound = m.objBound
    print("Limitante inferior:", lower_bound)
    up_bound = m.objBoundC
    print("Limitante superior:", up_bound)
    
    # Tentar obter e imprimir o valor ótimo
    optimal_value = m.objVal
    print("\nValor da solução ótima:", int(optimal_value))  # ou round(optimal_value)
    
except AttributeError as e:
    print("\nNão foi possível obter o valor ótimo:", str(e))
    
    # Calcular gap apenas se os bound estiverem disponíveis
    if 'up_bound' in locals() and 'lower_bound' in locals() and up_bound != 0:
        gap = 100 * (up_bound - lower_bound) / up_bound
        print(f"Gap de otimalidade: {gap:.2f}%")
    else:
        print("Não foi possível calcular o gap de otimalidade.")

# %%
m.update()

# %%
def gerar_arquivo():
    nome_arquivo = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\output_01.txt'
    with open(nome_arquivo, 'w') as f:
        # Verificar o status do modelo para garantir que a solução é ótima
        if m.status in [gp.GRB.OPTIMAL, 9]:
            
            # Imprimir os valores das variáveis de decisão relacionadas ao estoque nos CDs
            linha = f"Valores das variáveis de estoque nos CDs:\n"
            f.write(linha)
            for i in CDs:
                if i == 'CD_1':
                    for t in T:
                        linha = f"scrd[{i},{t}] = {scrd[i,t].X}\n"  # qestcd substituído por scrd
                        f.write(linha)

                    for t in T:
                        linha = f"scrf[{i},{t}] = {scrf[i,t].X}\n"  # qestcf substituído por scrf
                        f.write(linha)
                    
                    for t in T:
                        linha = f"scbp[{i},{t}] = {scbp[i,t].X}\n"  # qestcg substituído por scbp
                        f.write(linha)

                    for t in T:
                        linha = f"scbt[{i},{t}] = {scbt[i,t].X}\n"  # qestcg substituído por scbp
                        f.write(linha)
                    
                    for t in T:
                        linha = f"sc[{i},{t}] = {sc[i,t].X}\n"  # qestc substituído por sc
                        f.write(linha)

            # Imprimir os valores das variáveis de quantidade de envio dos CDs para os PAs
            linha = f"\nValores das variáveis de quantidade de envio dos CDs para os PAs:\n"
            f.write(linha)
            for t in T:
                for i in CDs:
                    for p in PAs:
                        linha = f"qap[{i},{p},{t}] = {qap[i,p,t].X}\n"  # qcp substituído por qap
                        f.write(linha)
            for t in T:
                for i in CDs:
                    for p in PAs:
                        linha = f"qrc[{p},{i},{t}] = {qrc[p,i,t].X}\n"  # qpc substituído por qrc
                        f.write(linha)
            for t in T:
                for i in CDs:
                    for d in Ds:
                        linha = f"qrd[{i},{d},{t}] = {qrd[i,d,t].X}\n"  # qcd substituído por qrd
                        f.write(linha)
            for t in T:
                for i in CDs:
                    for l in Fs:
                        linha = f"qrf[{i},{l},{t}] = {qrf[i,l,t].X}\n"  # qcf substituído por qrf
                        f.write(linha)
            for t in T:
                for i in CDs:
                    for l in Fs:
                        linha = f"qac[{l},{i},{t}] = {qac[l,i,t].X}\n"  # qfc substituído por qac
                        f.write(linha)

            # Imprimir os valores das variáveis de operação dos CDs
            linha = f"\nValores das variáveis de operação dos CDs:\n"
            f.write(linha)
            for t in T:
                for i in CDs:
                    linha = f"oc[{i},{t}] = {oc[i,t].X}\n"
                    f.write(linha)
                for i in CDs:
                    linha = f"z[{i},{t}] = {z[i,t].X}\n"
                    f.write(linha)
            
            # Imprimir os valores das variáveis de atendimento dos CDs aos PAs
            linha = f"\nValores das variáveis de atendimento dos CDs aos PAs:\n"
            f.write(linha)
            for t in T:
                for i in CDs:
                    for p in PAs:
                        if bcp[i,p,t].X == 1:
                            linha = f"bcp[{i},{p},{t}] = {bcp[i,p,t].X}\n"
                            f.write(linha)

            # Imprimir os valores das variáveis de estoque de Fabricante
            linha = f"\nValores das variáveis de operação dos Fabs:\n"
            f.write(linha)
            for l in Fs:
                for t in T:
                    linha = f"sf[{l},{t}] = {sf[l,t].X}\n"  # qestf substituído por sf
                    f.write(linha)
                for t in T:
                    linha = f"sfrf[{l},{t}] = {sfrf[l,t].X}\n"  # qestfx substituído por sfrf
                    f.write(linha)
                for t in T:
                    linha = f"sfbc[{l},{t}] = {sfbc[l,t].X}\n"  # qestfc substituído por sfbc
                    f.write(linha)

            linha = f"\nValores das variáveis de atendimento dos Fabs:\n"
            f.write(linha)
            for t in T:
                for l in Fs:
                    for i in CDs:
                        linha = f"qac[{l},{i},{t}] = {qac[l,i,t].X}\n"  # qfc substituído por qac
                        f.write(linha)
            for t in T:
                    for i in CDs:
                        linha = f"qfra[{l},{t}] = {qfra[l,t].X}\n"  # qfxg substituído por qfra
                        f.write(linha)

            # Imprimir os valores das variáveis de estoque de PA
            linha = f"\nValores das variáveis de operação dos PAs:\n"
            f.write(linha)
            for p in PAs:
                for t in T:
                    linha = f"sp[{p},{t}] = {sp[p,t].X}\n"  # qestp substituído por sp
                    f.write(linha)
                for t in T:
                    linha = f"sprc[{p},{t}] = {sprc[p,t].X}\n"  # qestpx substituído por sprc
                    f.write(linha)
                for t in T:
                    linha = f"spbj[{p},{t}] = {spbj[p,t].X}\n"  # qestpg substituído por spbj
                    f.write(linha)

            linha = f"\nValores das variáveis de atendimento dos PAs:\n"
            f.write(linha)
            for t in T:
                for p in PAs:
                    for i in CDs:
                        linha = f"qrc[{p},{i},{t}] = {qrc[p,i,t].X}\n"  # qpc substituído por qrc
                        f.write(linha)
            for t in T:
                for p in PAs:
                    for j in ZD:
                        linha = f"qaj[{p},{j},{t}] = {qaj[p,j,t].X}\n"  # qpj substituído por qaj
                        f.write(linha)
            for t in T:
                for p in PAs:
                    for j in ZD:
                        linha = f"qrp[{j},{p},{t}] = {qrp[j,p,t].X}\n"  # qjp substituído por qrp
                        f.write(linha)
            for t in T:
                for p in PAs:
                    for j in ZD:
                        linha = f"bpj[{p},{j},{t}] = {bpj[p,j,t].X}\n"
                        f.write(linha)

        else:
            print("Solução não encontrada ou não ótima. Status do modelo:", m.status)

try:
    gerar_arquivo()
except Exception as e:
    print("Erro ao gerar o arquivo:", str(e))

# %%
import os

# Certifique-se de que o diretório existe
output_dir = rf'C:\Users\rpafe\Desktop\lp'
os.makedirs(output_dir, exist_ok=True)
y = os.path.join(output_dir, 'model.ilp')
x = os.path.join(output_dir, 'model.lp')

# %%
import gurobipy as gp
import time

iss = []

# Marca o tempo inicial
start_time = time.time()
tempo_limite = 600  # 10 minutos em segundos

total_etapas = 5
for i in range(total_etapas):
    # Se já passou do tempo limite, interrompe o loop
    if time.time() - start_time > tempo_limite:
        print("\n⏰ Tempo limite de 10 minutos atingido. Encerrando o processo no estado atual...")
        break

    time.sleep(0.5)  # Simula algum processamento em cada etapa

    if i == 0 and m.status == gp.GRB.OPTIMAL:
        print("O modelo foi resolvido com sucesso!")
        print(f"Custo total: {m.objVal}")
    elif i == 1 and m.status == gp.GRB.INFEASIBLE:
        v_print = "O modelo é inviável."
        print("O modelo é inviável.")
        m.computeIIS()  # Identificar restrições inconsistentes
        m.write(y)  # Escrever modelo para análise
        m.write(x)
        print("\nMensagens do Gurobi:")
        for constr in m.getConstrs():
            if constr.IISConstr:
                iss.append(constr.ConstrName)
                print(f"Restrição IIS: {constr.ConstrName}")
        for var in m.getVars():
            if var.IISLB > 0 or var.IISUB > 0:
                print(f"Variável IIS: {var.VarName}, LB: {var.IISLB}, UB: {var.IISUB}")
    elif i == 2 and m.status == gp.GRB.UNBOUNDED:
        print("O modelo é ilimitado.")
    elif i == 3 and m.status not in [gp.GRB.OPTIMAL, gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED]:
        print("O modelo não pôde ser otimizado por um motivo desconhecido.")
    elif i == 4:
        print("Processo de verificação concluído.")

print("\n✅ Processo finalizado (completo ou por timeout).")


# %%
nome_arquivo = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\output_problemas.txt'

def verifica_restricoes(prefix):
    lista_restr = []
    with open(nome_arquivo, 'w') as f:
        for each_pref in prefix:
            for i in range(1, 999):
                formato = f'{i:03}'
                str_constr = each_pref + formato
                try:
                    for each_iss in iss:
                        str_var = str(globals()[str_constr])
                        if f'{each_iss}>' in str_var:
                            linha = f"{str_constr} - {each_iss} - problema\n"
                            # linha2 = f'{str_var}\n'
                            f.write(linha)
                            # f.write(linha2)
                            lista_restr.append(each_iss)
                            #print(str_constr, ' - ', each_iss, '- problema')
                except KeyError:
                    pass
                    # linha = f"{str_constr} - {each_iss} - não existe\n"
                    # f.write(linha)
                    #print(str_constr, ' - não existe')
                    # break
    return lista_restr


prefix = ['p', 'd', 'c', 'f']
lista_restr = verifica_restricoes(prefix)

# %%
# restricao_desejada = 'R312'
# chave_correspondente = None

# # Iterar sobre os itens do tupledict
# for chave, restricao in p020.items():
#     if restricao_desejada in str(restricao):
#         chave_correspondente = chave
#         break

# # Verificar se a chave foi encontrada
# if chave_correspondente is not None:
#     print(f"A chave correspondente à restrição {restricao_desejada} é {chave_correspondente}")
# else:
#     print(f"A restrição {restricao_desejada} não foi encontrada no tupledict")

# %%
import tkinter as tk
from tkinter import messagebox
import pygetwindow as gw

if 'v_print' in locals():
    file_path = "D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\output_problemas.txt"
    with open(file_path, 'r') as file:
    # Lê o conteúdo do arquivo
        content = file.read()
        os.startfile(file_path)

    # # Função para minimizar o VSCode
    # def minimize_vscode():
    #     vscode_windows = gw.getWindowsWithTitle('Visual Studio Code')
    #     if vscode_windows:
    #         vscode_windows[0].minimize()

    # # Função para maximizar o VSCode
    # def maximize_vscode():
    #     vscode_windows = gw.getWindowsWithTitle('Visual Studio Code')
    #     if vscode_windows:
    #         vscode_windows[0].maximize()

    # # Função para mostrar o pop-up
    # def show_popup():
    #     minimize_vscode()
    #     messagebox.showinfo(v_print, p002)
    #     maximize_vscode()

    # Mostrar o pop-up diretamente
    # show_popup()


# %%
import pandas as pd
import numpy as np

# Inicializando listas para armazenar as informações
refs = []
variaveis = []
aberto = []
ind_aberto = []
z_w = []
b_rota = []
tempo = []
origens = []
destinos = []
valores = []

# Preencher os dados com base nas variáveis calculadas no Gurobi
for t in T:
    for l in Fs:
        for i in CDs:
            if oc[i, t].X == 1:
                # Armazenar variáveis de fábrica para CDs
                refs.append('Fabrica')
                variaveis.append(f'qac[{l},{i},{t}]')
                aberto.append(np.nan)
                ind_aberto.append(np.nan)
                z_w.append(np.nan)
                b_rota.append(np.nan)
                tempo.append(t)
                origens.append(l)
                destinos.append(i)
                valores.append(qac[l, i, t].X)

        # Outras variáveis de fábrica
        refs.append('Fabrica')
        variaveis.append(f'qfra[{l},{t}]')
        aberto.append(np.nan)
        ind_aberto.append(np.nan)
        z_w.append(np.nan)
        b_rota.append(np.nan)
        tempo.append(t)
        origens.append(l)
        destinos.append(np.nan)
        valores.append(qfra[l, t].X)

        # Outras variáveis de fábrica
        refs.append('Fabrica')
        variaveis.append(f'qrf[{i},{l},{t}]')
        aberto.append(np.nan)
        ind_aberto.append(np.nan)
        z_w.append(np.nan)
        b_rota.append(np.nan)
        tempo.append(t)
        origens.append(l)
        destinos.append(np.nan)
        valores.append(qrf[i, l, t].X)

        # Estoque nas fábricas
        refs.append('Fabrica')
        variaveis.append(f'sf[{l},{t}]')
        aberto.append(np.nan)
        ind_aberto.append(np.nan)
        z_w.append(np.nan)
        b_rota.append(np.nan)
        tempo.append(t)
        origens.append(l)
        destinos.append(None)
        valores.append(sf[l, t].X)

        # Estoque nas fábricas
        refs.append('Fabrica')
        variaveis.append(f'sfrf[{l},{t}]')
        aberto.append(np.nan)
        ind_aberto.append(np.nan)
        z_w.append(np.nan)
        b_rota.append(np.nan)
        tempo.append(t)
        origens.append(l)
        destinos.append(None)
        valores.append(sfrf[l, t].X)

        # Estoque nas fábricas
        refs.append('Fabrica')
        variaveis.append(f'sfbc[{l},{t}]')
        aberto.append(np.nan)
        ind_aberto.append(np.nan)
        z_w.append(np.nan)
        b_rota.append(np.nan)
        tempo.append(t)
        origens.append(l)
        destinos.append(None)
        valores.append(sfbc[l, t].X)

    for i in CDs:
        if oc[i, t].X == 1:
            for p in PAs:
                # Armazenar variáveis de CDs
                refs.append('CD')
                variaveis.append(f'qap[{i},{p},{t}]')
                aberto.append(oc[i, t].X)
                ind_aberto.append(bcp[i, p, t].X)
                z_w.append(z[i, t].X)
                b_rota.append(bcp[i, p, t].X)
                tempo.append(t)
                origens.append(i)
                destinos.append(p)
                valores.append(qap[i, p, t].X)

            for d in Ds:
                refs.append('CD')
                variaveis.append(f'qrd[{i},{d},{t}]')
                aberto.append(oc[i, t].X)
                ind_aberto.append(np.nan)
                z_w.append(z[i, t].X)
                b_rota.append(1)
                tempo.append(t)
                origens.append(i)
                destinos.append(d)
                valores.append(qrd[i, d, t].X)

            for l in Fs:
                refs.append('CD')
                variaveis.append(f'qrf[{i},{l},{t}]')
                aberto.append(oc[i, t].X)
                ind_aberto.append(np.nan)
                z_w.append(z[i, t].X)
                b_rota.append(1)
                tempo.append(t)
                origens.append(i)
                destinos.append(l)
                valores.append(qrf[i, l, t].X)

            refs.append('CD')
            variaveis.append(f'sc[{i},{t}]')
            aberto.append(oc[i, t].X)
            ind_aberto.append(np.nan)
            z_w.append(z[i, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(i)
            destinos.append(np.nan)
            valores.append(sc[i, t].X)

            refs.append('CD')
            variaveis.append(f'scrf[{i},{t}]')
            aberto.append(oc[i, t].X)
            ind_aberto.append(np.nan)
            z_w.append(z[i, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(i)
            destinos.append(np.nan)
            valores.append(scrf[i, t].X)

            refs.append('CD')
            variaveis.append(f'scrd[{i},{t}]')
            aberto.append(oc[i, t].X)
            ind_aberto.append(np.nan)
            z_w.append(z[i, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(i)
            destinos.append(np.nan)
            valores.append(scrd[i, t].X)

            refs.append('CD')
            variaveis.append(f'scbp[{i},{t}]')
            aberto.append(oc[i, t].X)
            ind_aberto.append(np.nan)
            z_w.append(z[i, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(i)
            destinos.append(np.nan)
            valores.append(scbp[i, t].X)

            refs.append('CD')
            variaveis.append(f'scbt[{i},{t}]')
            aberto.append(oc[i, t].X)
            ind_aberto.append(np.nan)
            z_w.append(z[i, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(i)
            destinos.append(np.nan)
            valores.append(scbt[i, t].X)

    for p in PAs:
        if op[p, t].X == 1:
            for i in CDs:
                if oc[i, t].X == 1:
                    refs.append('PA')
                    variaveis.append(f'qrc[{p},{i},{t}]')
                    aberto.append(op[p, t].X)
                    ind_aberto.append(bcp[i, p, t].X)
                    z_w.append(w[p, t].X)
                    b_rota.append(bcp[i, p, t].X)
                    tempo.append(t)
                    origens.append(p)
                    destinos.append(i)
                    valores.append(qrc[p, i, t].X)

            for j in ZD:
                refs.append('PA')
                variaveis.append(f'qaj[{p},{j},{t}]')
                aberto.append(op[p, t].X)
                ind_aberto.append(np.nan)
                z_w.append(w[p, t].X)
                b_rota.append(bpj[p, j, t].X)
                tempo.append(t)
                origens.append(p)
                destinos.append(j)
                valores.append(qaj[p, j, t].X)

                refs.append('PA')
                variaveis.append(f'qrp[{j},{p},{t}]')
                aberto.append(op[p, t].X)
                ind_aberto.append(np.nan)
                z_w.append(w[p, t].X)
                b_rota.append(bpj[p, j, t].X)
                tempo.append(t)
                origens.append(p)
                destinos.append(j)
                valores.append(qrp[j, p, t].X)

            refs.append('PA')
            variaveis.append(f'sp[{p},{t}]')
            aberto.append(op[p, t].X)
            ind_aberto.append(np.nan)
            z_w.append(w[p, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(p)
            destinos.append(np.nan)
            valores.append(sp[p, t].X)

            refs.append('PA')
            variaveis.append(f'sprc[{p},{t}]')
            aberto.append(op[p, t].X)
            ind_aberto.append(np.nan)
            z_w.append(w[p, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(p)
            destinos.append(np.nan)
            valores.append(sprc[p, t].X)

            refs.append('PA')
            variaveis.append(f'spbj[{p},{t}]')
            aberto.append(op[p, t].X)
            ind_aberto.append(np.nan)
            z_w.append(w[p, t].X)
            b_rota.append(np.nan)
            tempo.append(t)
            origens.append(p)
            destinos.append(np.nan)
            valores.append(spbj[p, t].X)

# Criando o DataFrame
df = pd.DataFrame({
    'Ref': refs,
    'Variavel': variaveis,
    'Aberto': aberto,
    'z_w': z_w,
    'b_rota': b_rota,
    'Tempo': tempo,
    'Origem': origens,
    'Destino': destinos,
    'Valor': valores
})

df['Abrv Var'] = df['Variavel'].str.split('[').str[0]
df = df[['Ref', 'Variavel', 'Valor', 'Tempo', 'Aberto', 'z_w', 'b_rota', 'Origem', 'Destino', 'Abrv Var']]
df = df.fillna('')

# %%
dir = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Dados Gerados Mestrado\Auxiliar'

df_cluster_cds_cities = pd.read_csv(os.path.join(dir, 'df_cluster_cds_cities.csv')) # OK
df_cluster_pa_cities = pd.read_csv(os.path.join(dir, 'df_cluster_pa_cities.csv')) # OK

df_cluster_cds_cities = df_cluster_cds_cities.rename(columns={'CD': 'identificador', 'cidade': 'nome_cidade', \
                                                              'latitude_cidade':'lat_origem', 'longitude_cidade':'lon_origem'})
df_cluster_pa_cities = df_cluster_pa_cities.rename(columns={'PA': 'identificador', 'nome_cidade': 'nome_cidade', \
                                                            'latitude_cidade':'lat_origem', 'longitude_cidade':'lon_origem'})

df_cities = pd.concat([df_cluster_cds_cities[['identificador', 'uf', 'nome_cidade', 'lat_origem', 'lon_origem']], 
                       df_cluster_pa_cities[['identificador', 'uf', 'nome_cidade', 'lat_origem', 'lon_origem']]], ignore_index=True)

df = df.merge(df_cities, left_on='Origem', right_on='identificador', how='left')
df = df.drop(columns=['identificador'])
df = df.rename(columns={'uf':'uf_origem', 'nome_cidade':'cid_origem'})
df = df.fillna('')

# %%
# Exibir o DataFrame
raw_dir = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script'
n = rf'{raw_dir}\Results\results.csv'

df.to_csv(n, encoding='latin-1',index=False)

# %%
import pandas as pd
import gurobipy as gp

# Assumindo que 'm' é o seu modelo Gurobi já resolvido
# m = gp.Model("MeuModelo")
# ... (código para construir e resolver o modelo)

# Lista dos nomes base das variáveis que aparecem na função objetivo
# Adicione ou remova nomes conforme necessário, se a sua função objetivo mudar
objective_var_names = [
    'z', 'oc', 'w', 'op', 'qac', 'qrf', 'qap', 'qrc', 'qrd', 'qaj', 'qrp'
]

data = []
for v in m.getVars():
    var_name = v.varName
    
    # Extrair o nome base da variável (antes do primeiro '[')
    base_name = var_name.split('[')[0]

    # Verificar se a variável está na lista de variáveis da função objetivo
    if base_name in objective_var_names:
        parts = var_name.split('[')
        name = parts[0]
        indices = None
        if len(parts) > 1:
            indices = parts[1].replace(']', '')

        # Extrair o 'tempo' se estiver presente nos índices
        time_index = None
        if indices:
            index_parts = indices.split(',')
            # A lógica para identificar o índice de tempo pode variar.
            # Aqui, estamos assumindo que o índice de tempo é o último para a maioria das variáveis
            # e tentamos converter para int.
            try:
                time_index = int(index_parts[-1].strip())
            except ValueError:
                # Se o último índice não for numérico, talvez seja um outro tipo de índice ou não há tempo explícito
                time_index = None # Ou você pode definir uma string 'N/A'

        data.append({
            'variavel': name,
            'tempo': time_index,
            'indices': indices,
            'valor': v.X
        })

# Criar um DataFrame pandas
df = pd.DataFrame(data)

# Salvar em um arquivo CSV
df.to_csv(rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\resultados_variaveis_fo.csv', index=False)

print("Resultados das variáveis da função objetivo exportados para 'resultados_variaveis_fo.csv'")

# %%
# constraints_ls = [i for i in m.getConstrs() if 'xyz_upper_bound' not in i.constrname if 'xyz_lower_bound' not in i.constrname]
# relaxed_model = m.feasRelaxS(
#                 relaxobjtype=0, minrelax=False, vrelax=False, crelax=True
#             )

# %%
# nome_arquivo = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results\output_revisar.txt'

# with open(nome_arquivo, 'w') as f:
#     for each in lista_restr:
#         f.write('\n')
#         f.write(f'Restrição = {each}.\n')
#         each = m.getVarByName(each)
#         for chave, valor in p010.items():
#             linha = f'Chave: {chave}, Valor: {valor}\n'
#             f.write(linha)


