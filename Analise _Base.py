# %%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# %%
dir_path = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Results'
name_path = 'results.csv'
complete_path = os.path.join(dir_path, name_path)

df_res = pd.read_csv(complete_path, encoding='latin-1')
df_res = df_res.loc[df_res['Valor'] != 0, :]
df_res.loc[:, 'Tempo'] = df_res.loc[:, 'Tempo'] + 1

# %%
dir = rf'D:\OneDrive\Documentos\Materiais didáticos\MECAI\Dissertação\script\Dados Gerados Mestrado\Auxiliar'

# df_cluster_cds_cities = pd.read_csv(os.path.join(dir, 'df_cluster_cds_cities.csv'))
# df_cluster_pa_cities = pd.read_csv(os.path.join(dir, 'df_cluster_pa_cities.csv'))
df_dist_CD_PA = pd.read_csv(os.path.join(dir, 'df_dist_CD_PA.csv'))
df_dist_ZD_PA = pd.read_csv(os.path.join(dir, 'df_dist_ZD_PA.csv'))
df_instalacao = pd.read_csv(os.path.join(dir, 'df_instalacao.csv'))
df_desinstalacao = pd.read_csv(os.path.join(dir, 'df_desinstalacao.csv'))
df_dist_Fab_CD = pd.read_csv(os.path.join(dir, 'df_dist_Fab_CD.csv'))
df_dist_CD_Desc = pd.read_csv(os.path.join(dir, 'df_dist_CD_Desc.csv'))
# df_regiao_pa_zd = pd.read_csv(os.path.join(dir, 'df_regiao_pa_zd.csv'))

df_estq_inicial = pd.read_csv(os.path.join(dir, 'df_estq_inicial.csv'))
df_estq_inicial = df_estq_inicial.loc[:, ['PA', 'lim_estq_in_pa', 'Restr_Rec', 'Restr_Env', 'custo_por_faixa', 'custo_man_por_faixa']]

# %%
def pivotar_demanda_mensal(df, colunas_valor, name_col, name_col_pivot, mapa=None):
    """
    Função para pivotar o DataFrame de instalação, transformando as colunas de demanda mensal
    em uma coluna única com os meses representados por números de 0 a 11.
    
    Parâmetros:
    df_instalacao (DataFrame): DataFrame contendo as colunas de demanda mensal.
    
    Retorna:
    DataFrame: DataFrame pivotado com colunas 'ZD', 'mes' e 'demanda'.
    """
 
    # Aplicando melt
    df = df[[name_col] + colunas_valor].melt(
        id_vars=name_col,
        value_vars=colunas_valor,
        var_name= name_col_pivot,
        value_name='demanda'
    )

    if mapa is not None:
        df[name_col_pivot] = df[name_col_pivot].replace(mapa)

    return df

# Lista com os nomes originais das colunas de demanda mensal
colunas_demanda = [
    'demanda_normal_Jan', 'demanda_normal_Fev', 'demanda_normal_Mar',
    'demanda_normal_Abr', 'demanda_normal_Mai', 'demanda_normal_Jun',
    'demanda_normal_Jul', 'demanda_normal_Ago', 'demanda_normal_Set',
    'demanda_normal_Out', 'demanda_normal_Nov', 'demanda_normal_Dez'
]

# Dicionário para mapear os meses para valores de 1 a 12
mapa_meses = {
    'demanda_normal_Jan': 1, 'demanda_normal_Fev': 2, 'demanda_normal_Mar': 3,
    'demanda_normal_Abr': 4, 'demanda_normal_Mai': 5, 'demanda_normal_Jun': 6,
    'demanda_normal_Jul': 7, 'demanda_normal_Ago': 8, 'demanda_normal_Set': 9,
    'demanda_normal_Out': 10, 'demanda_normal_Nov': 11, 'demanda_normal_Dez': 12
}

# Substituir nomes dos meses pelos índices numéricos
df_inst_pivot = pivotar_demanda_mensal(df_instalacao, colunas_demanda, 'ZD', 'mes', mapa_meses)
df_dinst_pivot = pivotar_demanda_mensal(df_desinstalacao, colunas_demanda, 'ZD', 'mes', mapa_meses)

# %%
import pandas as pd

def union_dfs_with_different_columns(list_dfs):
    """
    Realiza a união (concatenação) de quatro DataFrames, cada um com 3 colunas,
    mesmo que os nomes das colunas sejam diferentes.

    Args:
        df1 (pd.DataFrame): O primeiro DataFrame.
        df2 (pd.DataFrame): O segundo DataFrame.
        df3 (pd.DataFrame): O terceiro DataFrame.
        df4 (pd.DataFrame): O quarto DataFrame.

    Returns:
        pd.DataFrame: Um novo DataFrame resultante da união dos quatro DataFrames,
                      com as colunas renomeadas para 'coluna_1', 'coluna_2', 'coluna_3'.
    """
    dfs = [list_dfs[i] for i in range(len(list_dfs))]
    unified_dfs = []

    for i, df in enumerate(dfs):
        if df.shape[1] != 3:
            raise ValueError(f"DataFrame {i+1} não possui 3 colunas. Possui {df.shape[1]}.")

        # Renomeia as colunas para um padrão comum
        df_renamed = df.copy()
        df_renamed.columns = ['Origem_Dist', 'Destino_Dist', 'Distancia']
        unified_dfs.append(df_renamed)

    # Concatena todos os DataFrames renomeados
    union_df = pd.concat(unified_dfs, ignore_index=True)

    return union_df

# %%
colunas_cd_pas = [each for each in df_dist_CD_PA.columns if each[:3] == 'PA_']
df_dist_CD_PA_pivot = pivotar_demanda_mensal(df_dist_CD_PA, colunas_cd_pas, 'CD', 'PA')

colunas_pas_zd = [each for each in df_dist_ZD_PA.columns if each[:3] == 'PA_']
df_dist_ZD_PA_pivot = pivotar_demanda_mensal(df_dist_ZD_PA, colunas_pas_zd, 'ZD', 'PA')

colunas_cd_fab = [each for each in df_dist_Fab_CD.columns if each[:3] == 'CD_']
df_dist_Fab_CD_pivot = pivotar_demanda_mensal(df_dist_Fab_CD, colunas_cd_fab, 'FAB', 'CD')

colunas_cd_desc = [each for each in df_dist_CD_Desc.columns if each[:3] == 'CD_']
df_dist_CD_Desc_pivot = pivotar_demanda_mensal(df_dist_CD_Desc, colunas_cd_desc, 'DESC', 'CD')

list_dfs = [df_dist_CD_PA_pivot, df_dist_ZD_PA_pivot, df_dist_Fab_CD_pivot, df_dist_CD_Desc_pivot]
df_dist = union_dfs_with_different_columns(list_dfs)

# %%
def left_join(df1, df2, left_col, right_col, prefix, how='left'):
    # Descobrir colunas que não são chaves
    non_key_cols = [col for col in df2.columns]
        
    # Fazer o merge normalmente
    df = df1.merge(df2, left_on=left_col, right_on=right_col, how=how)

    # Renomear colunas do df2 com prefixo
    df = df.copy().rename(columns={col: f"{prefix}{col}" for col in non_key_cols})
    
    # Remover a coluna de chave do df2
    right_list = [f"{prefix}{right_col[i]}" for i in range(len(right_col)) if f"{prefix}{right_col[i]}" in df.columns]
    df = df.drop(columns=right_list, errors='ignore')

    return df

# %%
df_res = left_join(df_res, df_inst_pivot, ['Destino', 'Tempo'], ['ZD', 'mes'], 'p_inst_')
df_res = left_join(df_res, df_dinst_pivot, ['Destino', 'Tempo'], ['ZD', 'mes'], 'p_dinst_')

# %%
df_res = left_join(df_res, df_dist, ['Origem', 'Destino'], ['Origem_Dist', 'Destino_Dist'], 'p_')
df_res = left_join(df_res, df_dist, ['Origem', 'Destino'], ['Destino_Dist', 'Origem_Dist'], 'p_aux_')
df_res['p_Distancia'] = df_res['p_Distancia'].fillna(df_res['p_aux_Distancia'])
df_res = df_res.drop(columns=['p_aux_Distancia'])

# %%
df_res = left_join(df_res, df_estq_inicial, ['Origem'], ['PA'], 'p_ori_pa_')
df_res = left_join(df_res, df_estq_inicial, ['Destino'], ['PA'], 'p_dest_pa_')

# %%
def substituir_colunas(df, list_col1, list_col2):
    for col1, col2 in zip(list_col1, list_col2):
        if col1 in df.columns and col2 in df.columns:
            df[col1] = df[col1].fillna(df[col2])
            df = df.drop(columns=[col2])
    return df

df_res = substituir_colunas(df_res, ['p_ori_pa_custo_man_por_faixa', 'p_ori_pa_custo_por_faixa', 'p_ori_pa_lim_estq_in_pa', 'p_ori_pa_Restr_Rec', 'p_ori_pa_Restr_Env'], \
        ['p_dest_pa_custo_man_por_faixa', 'p_dest_pa_custo_por_faixa', 'p_dest_pa_lim_estq_in_pa', 'p_dest_pa_Restr_Rec', 'p_dest_pa_Restr_Env'])

# %%
# Fabricantes
rng_estq_fab = [10000, 10002]
mrf = 9000
maf = 9000
ef = np.ceil(rng_estq_fab[1]*2.4)  # estf substituído por ef
rf = round(0.2, 1)

# CDs
num_cds_sel = 1
rng_estq_cd = [110000, 110002]
rng_rest_cd = [rng_estq_cd[0]*0.75, rng_estq_cd[1]*0.75]

ec = np.ceil(rng_estq_cd[1]*2.1)  # estc substituído por ec
min_estq_bom_cd = np.ceil(ec*0.2)
nc = num_cds_sel  # mc substituído por nc
rc = round(0.7, 1)

# Descartes
dr = 1000000
rd = round(1 - rc - rf, 1)


# %%
vi = 100000  # c_man_cd substituído por vi
hi = 3000000  # c_ab_cd substituído por hi
k = 0.087189

# %%
df_res['p_fab_estq_fab'] = rng_estq_fab[1]
df_res['p_fab_rest_env'] = maf
df_res['p_fab_rest_rec'] = mrf
df_res['p_ori_fab_lim_estq'] = ef

df_res['p_cd_estq_fab'] = rng_estq_cd[1]
df_res['p_cd_rest_env'] = rng_rest_cd[1]
df_res['p_cd_rest_rec'] = rng_rest_cd[1]
df_res['p_ori_cd_lim_estq'] = ec

# %%
df_res['p_custo_km'] = k
df_res['p_cd_custo_man'] = vi
df_res['p_cd_custo_ab'] = hi

# %%
def create_boolean_columns_generic(df, column_mappings, source_column='Abrv Var'):
    for new_column, values in column_mappings.items():
        df[new_column] = df[source_column].isin(values)
    return df

dados_mapeados_por_coluna = {
    'env_cd': ['qrd', 'qap', 'qrf'], 'rec_cd': ['qac', 'qrc'], 'env_pa': ['qaj', 'qrc'], 
    'rec_pa': ['qap', 'qrp'], 'env_fab': ['qac'], 'rec_fab': ['qrf']}

df_res = create_boolean_columns_generic(df_res.copy(), dados_mapeados_por_coluna)

df_res['estq'] = df_res['Abrv Var'].str.startswith('s')
df_res['movim'] = df_res['Abrv Var'].str.startswith('q')

# %%
# Opção 3: Função usando pandas vetorizado (mais eficiente)
def criar_colunas_pa_v3(df, col_origem='Origem', col_destino='Destino', 
                        prefixo='PA', nome_col1='PA', nome_col2='PA_Outros'):
    """
    Versão vetorizada usando pandas - mais eficiente para DataFrames grandes.
    
    Args:
        df: DataFrame
        col_origem: nome da coluna origem
        col_destino: nome da coluna destino
        prefixo: string para verificar no startswith
        nome_col1: nome da primeira coluna criada
        nome_col2: nome da segunda coluna criada
    
    Returns:
        DataFrame com as novas colunas
    """
    df = df.copy()
    
    # Máscaras para verificar onde está o prefixo
    mask_origem = df[col_origem].str.startswith(prefixo)
    mask_destino = df[col_destino].str.startswith(prefixo)
    
    # Condição: pelo menos uma das colunas deve ter o prefixo
    tem_prefixo = mask_origem | mask_destino
    
    # Primeira coluna: se origem tem prefixo, pega destino; senão pega origem
    df[nome_col1] = np.where(tem_prefixo, 
                            np.where(mask_origem, df[col_origem], df[col_destino]),
                            None)
    
    # Segunda coluna: se origem tem prefixo, pega origem; senão pega destino  
    df[nome_col2] = np.where(tem_prefixo,
                            np.where(mask_origem, df[col_destino], df[col_origem]),
                            None)
    
    return df

df_res = criar_colunas_pa_v3(df_res.copy(), col_origem='Origem', col_destino='Destino',
                             prefixo='PA', nome_col1='PA', nome_col2='PA_Outros')

# %%
# Opção 3: Versão em uma linha (corrigida)
mask = ((~df_res['PA'].isna()) & (~df_res['PA_Outros'].isna())) & (df_res['rec_pa']==True)
df_res.loc[mask, 'p_ori_pa_Restr_Rec_frac'] = (
    df_res.loc[mask].groupby(['Tempo', 'PA'])['p_ori_pa_Restr_Rec'].transform('max') / 
    df_res.loc[mask].groupby(['Tempo', 'PA'])['PA_Outros'].transform('count')
)
df_res.loc[mask, 'Valor_Rec_PA'] = df_res.loc[mask, 'Valor']

mask = ((~df_res['PA'].isna()) & (~df_res['PA_Outros'].isna())) & (df_res['env_pa']==True)
df_res.loc[mask, 'p_ori_pa_Restr_Env_frac'] = (
    df_res.loc[mask].groupby(['Tempo', 'PA'])['p_ori_pa_Restr_Env'].transform('max') / 
    df_res.loc[mask].groupby(['Tempo', 'PA'])['PA_Outros'].transform('count')
)
df_res.loc[mask, 'Valor_Env_PA'] = df_res.loc[mask, 'Valor']

mask = (df_res['estq']==True)
df_res.loc[mask, 'Estq_PA'] = df_res.loc[mask, 'Valor']

# %%
df_res['r_custo_ab_cd'] = df_res['p_cd_custo_ab'] * df_res['z_w']
df_res['r_custo_man_cd'] = df_res['p_cd_custo_man'] * df_res['Aberto']
df_res['r_custo_ab_pa'] = df_res['p_ori_pa_custo_por_faixa'] * df_res['z_w']
df_res['r_custo_man_pa'] = df_res['p_ori_pa_custo_man_por_faixa'] * df_res['Aberto']
df_res['r_custo_mov'] = df_res['Valor'] * df_res['p_Distancia'] * df_res['p_custo_km']

# %%
# df_res.to_csv(os.path.join(dir_path, "df_res_v3.csv"), encoding='latin-1', index=False)
# df_res.to_excel(os.path.join(dir_path, "df_res_v3.xlsx"), index=False)


