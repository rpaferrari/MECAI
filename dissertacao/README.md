# Dissertação de Mestrado — ICMC-USP (LaTeX)

Este diretório contém os códigos-fonte em LaTeX e o PDF compilado da dissertação de mestrado apresentada ao Instituto de Ciências Matemáticas e de Computação (ICMC-USP) pelo programa de pós-graduação **MECAI** (Mestrado Profissional em Matemática, Estatística e Computação Aplicadas à Indústria).

- **Título:** *Otimização da Cadeia de Suprimentos de Circuito Fechado de Adquirência por meio de um Modelo de Otimização Linear Inteira Mista*
- **Autor:** Rafael Pereira de Ávila Ferrari
- **Orientadora:** Profa. Dra. Maristela Oliveira dos Santos
- **Coorientadora:** Profa. Dra. Eli Angela Vitor Toso

---

## 📕 Documento Compilado

O PDF final compilado (86 páginas, formato ABNT) está disponível diretamente em:
👉 [**`thesis.pdf`**](thesis.pdf)

---

## ⚙️ Instruções de Compilação

Para compilar este documento a partir dos fontes, certifique-se de possuir uma distribuição TeX instalada (ex.: MiKTeX ou TeX Live).

> **Atenção:** O projeto **deve ser compilado utilizando XeLaTeX** (ou LuaLaTeX), pois ele faz uso do pacote `fontspec` e das fontes TrueType presentes em `packages/fonts/` para a formatação oficial da capa e folhas de rosto do ICMC.

### Ciclo de Compilação Completo:

```powershell
# 1ª passagem XeLaTeX
xelatex -interaction=nonstopmode thesis.tex

# Processamento da bibliografia ABNT
bibtex thesis

# Processamento da lista de siglas e símbolos
makeindex thesis.nlo -s nomencl.ist -o thesis.nls

# 2ª e 3ª passagens para consolidação de referências cruzadas e sumário
xelatex -interaction=nonstopmode thesis.tex
xelatex -interaction=nonstopmode thesis.tex
```

---

## 📁 Estrutura de Arquivos

- **`thesis.tex`**: Arquivo mestre que importa todos os capítulos e configurações.
- **`thesis.pdf`**: Documento compilado final.
- **`references.bib`**: Base de dados bibliográfica BibTeX (padrão ABNT).
- **`tex/`**: Contém todos os capítulos divididos modularmente:
  - `01_introducao.tex`: Introdução, contexto da indústria e objetivos.
  - `02_revisao.tex`: Revisão da literatura em logística reversa e localização de facilidades.
  - `03_def_prob_mod.tex`: Definição do problema e formulação matemática em PLIM.
  - `04_met_e_prep.tex`: Metodologia, dados e construção dos cenários.
  - `05_exp_comp.tex`: Experimentos computacionais e análise dos resultados.
  - `06_conc.tex`: Conclusões e direcionamentos futuros.
  - `pre-textual/`: Resumo, Abstract, Agradecimentos, Epígrafe e Ficha Catalográfica.
  - `appendix/` e `annex/`: Apêndices e anexos.
- **`packages/`**: Classe `icmc.cls`, arquivos de estilo (`abntexalfenglish.bst`) e fontes.
- **`images/`**: Imagens, fluxogramas logísticos e diagramas.
