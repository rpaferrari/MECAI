# MECAI - Mestrado Profissional em Matemática, Estatística e Computação Aplicadas à Indústria

> **Dissertação de Mestrado — Instituto de Ciências Matemáticas e de Computação (ICMC-USP)**  
> **Título:** *Otimização da Cadeia de Suprimentos de Circuito Fechado de Adquirência por meio de um Modelo de Otimização Linear Inteira Mista*  
> **Autor:** Rafael Pereira de Ávila Ferrari  
> **Orientadora:** Profa. Dra. Maristela Oliveira dos Santos  
> **Coorientadora:** Profa. Dra. Eli Angela Vitor Toso  

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Gurobi Optimizer](https://img.shields.io/badge/Solver-Gurobi%20Optimizer-red.svg)](https://www.gurobi.com/)
[![LaTeX](https://img.shields.io/badge/LaTeX-XeLaTeX-008080.svg?logo=latex&logoColor=white)](dissertacao/)
[![Institution](https://img.shields.io/badge/ICMC-USP-orange.svg)](https://www.icmc.usp.br/)
[![PDF](https://img.shields.io/badge/PDF-Dissertação%20Completa-brightgreen.svg)](dissertacao/thesis.pdf)

---

## 📄 Resumo do Trabalho

O mercado de adquirência é composto por empresas credenciadoras que fornecem serviços de transação eletrônica entre estabelecimentos comerciais (pontos de venda) e clientes, predominantemente por meio de terminais de pagamento (maquininhas POS). No modelo operacional dessas credenciadoras, a disponibilidade do parque de terminais é regida por rígidos Acordos de Nível de Serviço (*Service Level Agreement* - SLA), exigindo prazos de até 24 horas para reparo ou substituição de equipamentos com defeito.

Este trabalho propõe um modelo matemático de **Programação Linear Inteira Mista (PLIM)** para otimizar uma **Cadeia de Suprimentos de Circuito Fechado (*Closed-Loop Supply Chain*)** no setor de adquirência sob a ótica de logística reversa e economia circular. 

A malha logística engloba:
- **Fabricantes (Fornecedores)** de terminais novos;
- **Centros de Distribuição (CDs)** centrais de consolidação e reparo;
- **Postos Avançados (PAs)** para atendimento descentralizado e resposta rápida ao SLA;
- **Zonas de Demanda (ZDs)** com concentração de clientes/estabelecimentos;
- **Unidades de Descarte** e reciclagem de terminais em fim de vida útil (*end-of-life*).

O objetivo é minimizar o custo operacional total (instalação e custos fixos de CDs e PAs, estocagem, remanufatura e fretes diretos e reversos), garantindo capacidades de processamento e atendimento integral da demanda e SLAs.

---

## 🗂 Estrutura do Repositório

```text
MECAI/
├── dissertacao/                    # Código-fonte LaTeX e PDF da dissertação
│   ├── thesis.pdf                  # 📕 Documento final completo compilado (86 páginas)
│   ├── thesis.tex                  # Arquivo mestre LaTeX (XeLaTeX)
│   ├── references.bib              # Base de referências bibliográficas (BibTeX / ABNT)
│   ├── tex/                        # Capítulos da dissertação
│   │   ├── 01_introducao.tex       # Introdução e contexto do mercado
│   │   ├── 02_revisao.tex          # Revisão bibliográfica (Logística Reversa / PLIM)
│   │   ├── 03_def_prob_mod.tex     # Formulação matemática detalhada do modelo PLIM
│   │   ├── 04_met_e_prep.tex       # Metodologia e preparação de instâncias e dados
│   │   ├── 05_exp_comp.tex         # Experimentos computacionais e análise de resultados
│   │   ├── 06_conc.tex             # Conclusões e trabalhos futuros
│   │   └── pre-textual/            # Agradecimentos, dedicatória, resumo, abstract
│   ├── packages/                   # Classe ICMC-USP (icmc.cls), folhas de estilo e fontes
│   └── images/                     # Mapas, gráficos e figuras da dissertação
│
├── figuras/                        # Mapas geográficos e gráficos gerados nos experimentos
├── mestrado.py                     # Implementação do modelo PLIM em Python (Gurobi)
├── Analise.py                      # Rotinas de consolidação e métricas dos resultados
├── Analise _Base.py                # Análises comparativas de cenários
├── gerador_de_dados.ipynb          # Notebook de geração de instâncias e distribuições de demanda
├── Mestrado.ipynb                  # Notebook de execução dos experimentos e otimização
├── Mestrado_Analise.ipynb          # Visualização e gráficos das rotas e alocações
├── Mestrado_Analise_Base.ipynb     # Comparações de baseline
└── README.md                       # Apresentação do projeto e instruções de uso
```

---

## 🔬 Modelagem Matemática e Implementação

- **Linguagem:** Python 3.x
- **Modelagem & Solver:** [Gurobi Optimizer](https://www.gurobi.com/) via biblioteca `gurobipy`
- **Geoprocessamento e Mapas:** Folium, GeoPandas, Shapely
- **Processamento de Dados:** Pandas, NumPy
- **Escrita Acadêmica:** LaTeX com motor **XeLaTeX** (classe `packages/icmc.cls` com suporte a fontes TrueType Arial e normas ABNT via `abntex2`).

---

## 🚀 Como Executar

### 1. Modelos de Otimização e Experimentos (Python)

1. Clone o repositório:
   ```bash
   git clone https://github.com/rpaferrari/MECAI.git
   cd MECAI
   ```

2. Instale as dependências recomendadas:
   ```bash
   pip install gurobipy pandas numpy matplotlib folium openpyxl jupyter
   ```
   *(Nota: É necessária uma licença ativa do Gurobi Optimizer para resolução das instâncias).*

3. Execute o script principal ou os notebooks interativos:
   ```bash
   python mestrado.py
   # ou inicie o Jupyter
   jupyter lab
   ```

---

### 2. Compilação da Dissertação (LaTeX)

Para compilar o documento acadêmico a partir do código-fonte:

1. Acesse o diretório `dissertacao/`:
   ```bash
   cd dissertacao
   ```

2. Compile utilizando **XeLaTeX** e os utilitários de bibliografia e nomenclatura:
   ```bash
   xelatex thesis.tex
   bibtex thesis
   makeindex thesis.nlo -s nomencl.ist -o thesis.nls
   xelatex thesis.tex
   xelatex thesis.tex
   ```
   O documento compilado será gerado como [`dissertacao/thesis.pdf`](dissertacao/thesis.pdf).

---

## 📖 Como Citar

```bibtex
@mastersthesis{Ferrari2026Otimizacao,
  author  = {Ferrari, Rafael Pereira de {\'A}vila},
  title   = {Otimiza{\c{c}}{\~a}o da Cadeia de Suprimentos de Circuito Fechado de Adquir{\^e}ncia por meio de um Modelo de Otimiza{\c{c}}{\~a}o Linear Inteira Mista},
  school  = {Instituto de Ci{\^e}ncias Matem{\'a}ticas e de Computa{\c{c}}{\~a}o, Universidade de S{\~a}o Paulo (ICMC-USP)},
  year    = {2026},
  address = {S{\~a}o Carlos, SP},
  note    = {Disserta{\c{c}}{\~a}o de Mestrado Profissional em Matem{\'a}tica, Estat{\'\i}stica e Computa{\c{c}}{\~a}o Aplicadas {\`a} Ind{\'u}stria (MECAI)}
}
```

---

## 📜 Licença

Os códigos e modelos estão disponíveis sob os termos da licença deste repositório. Para detalhes, consulte o arquivo [LICENSE](dissertacao/LICENSE).
