# PROPOSTA DE PROJETO FINAL
**Disciplina:** CDD109 – Redes Neurais Artificiais e Deep Learning  
**Instituição:** Universidade Federal de Ouro Preto (UFOP) – Departamento de Computação  

---

### 1. Nome do Grupo / Integrantes
* Leonardo Isaac Silva Flores
* Eduardo Alexandre Sobreira

---

### 2. Base de Dados Escolhida
* **Título do Projeto:** Classificação de Vulnerabilidade Social por Setores Censitários para Dimensionamento da Capacidade de PSFs em Ouro Preto.
* **Fonte dos Dados:** Censo Demográfico do IBGE (Agregados por Setores Censitários / Plataforma SIDRA) integrado ao Atlas da Vulnerabilidade Social (IPEA / IVS) e aos limites territoriais das microáreas de PSFs de Ouro Preto.
* **Contexto e Motivação:** Na atenção primária do SUS (normatizada pela PNAB), a capacidade de atendimento de uma equipe de Saúde da Família (PSF) deve ser inversamente proporcional à vulnerabilidade social da comunidade: territórios com alto grau de vulnerabilidade demandam cuidados mais intensivos e visitas mais frequentes, devendo acolher um número menor de pacientes adscritos (máximo de 2.400), ao passo que áreas de baixo risco comportam até 4.000 pessoas. O presente projeto visa classificar via redes neurais o risco social das microáreas de Ouro Preto a partir de indicadores socioeconômicos reais do IBGE, superando o uso de dados aleatórios e consolidando a base de dados necessária para modelos de otimização logística de alocação de equipes.

#### Detalhamento Técnico da Base:
* **Número de amostras:**  
  A base de treinamento contará com aproximadamente **1.800 setores censitários**, englobando a Microrregião dos Inconfidentes e Central de MG (Ouro Preto, Mariana, Itabirito, Congonhas e Conselheiro Lafaiete) para assegurar volume estatístico adequado ao treinamento da rede neural. Em seguida, o modelo realizará a inferência e avaliação específica nas **60 microáreas** que delimitam a cobertura dos PSFs da cidade de Ouro Preto.
* **Quais são as variáveis de entrada e saída:**  
  * **Variáveis de Entrada (8 indicadores socioeconômicos do IBGE por setor):**
    1. *Renda:* Renda domiciliar per capita média do setor censitário;
    2. *Abastecimento:* Percentual de domicílios sem acesso à rede geral de distribuição de água;
    3. *Esgotamento:* Percentual de domicílios sem esgotamento sanitário adequado (rede geral ou fossa séptica);
    4. *Coleta de Lixo:* Percentual de domicílios sem coleta regular de lixo;
    5. *Educação:* Taxa de analfabetismo na população de 15 anos ou mais;
    6. *Dependência Demográfica:* Razão de dependência (proporção de crianças de 0 a 14 anos e idosos de 65+ em relação à população em idade ativa);
    7. *Composição Familiar:* Percentual de chefia familiar monomarental feminina com dependentes;
    8. *Densidade Habitacional:* Média de moradores por dormitório/cômodo.
  * **Variável de Saída (`Target` - 3 classes ordinais alinhadas à Escala de Risco Familiar de Coelho-Savassi / PNAB):**
    * **R1 – Menor Risco (Baixa Vulnerabilidade):** área estruturada; capacidade recomendada de até **4.000 pessoas por PSF** (até 750 por microárea/ACS);
    * **R2 – Médio Risco (Vulnerabilidade Moderada):** carência intermediária; capacidade recomendada de até **3.200 pessoas por PSF** (até 550 por microárea/ACS);
    * **R3 – Alto Risco (Alta Vulnerabilidade / Risco Máximo):** expressiva privação socioeconômica; capacidade limitada a **2.400 pessoas por PSF** (até 400 por microárea/ACS).
* **Tem dados faltantes (missing values)?**  
  **Sim.** Setores de uso misto, áreas rurais isoladas ou com menos de 5 domicílios particulares possuem atributos omitidos pelo IBGE por sigilo estatístico.  
  *Tratamento proposto:* Descarte de setores estritamente não residenciais (ex.: áreas industriais ou parques florestais) e imputação pela mediana municipal para dados pontuais faltantes.
* **Tem outliers?**  
  **Sim.** Setores de alto padrão aquisitivo (centro histórico e condomínios) e bolsões de extrema carência periférica geram caudas longas em renda e saneamento.  
  *Tratamento proposto:* Normalização robusta com `RobustScaler` (ou `StandardScaler`) e truncamento de extremos via percentis.
* **Natureza do problema?**  
  **Classificação Multiclasse Supervisionada**.
* **É balanceada?**  
  **Não.** Setores de médio risco (R2) concentram o maior contingente populacional, enquanto setores de risco muito alto (R3) ou muito baixo (R1) são menos numerosos.  
  *Tratamento proposto:* Ajuste de hiperparâmetros de penalização na função de perda (`class_weight`), com avaliação de desempenho através de **Macro F1-Score** e matriz de confusão multiclasse.

---

### 3. Qual modelo de rede neural pretende usar? (MLP, CNN, RNN)
* **Modelo:** **MLP (Multi-Layer Perceptron)** tabular profunda desenvolvida em **Keras / TensorFlow**.
* **Justificativa e Arquitetura Planejada:**  
  A relação entre indicadores de saneamento, renda e densidade domiciliar não é linear e apresenta forte interação cruzada. Uma MLP com conexões densas é a estrutura canônica da disciplina para extração desses padrões tabulares:
  1. **Camada de Entrada:** 8 neurônios correspondentes aos atributos socioeconômicos normalizados;
  2. **Camadas Ocultas:** 2 camadas densas (`Dense`) com 32 a 64 neurônios e função de ativação **ReLU**;
  3. **Mecanismos de Regularização:** Camadas de **Dropout** (taxa entre 0.2 e 0.3) e **BatchNormalization** após cada camada densa para estabilização do gradiente e prevenção de sobreajuste (*overfitting*);
  4. **Camada de Saída:** 3 neurônios com função de ativação **Softmax**;
  5. **Compilação:** Otimizador **Adam** e função de perda `categorical_crossentropy` (ou `sparse_categorical_crossentropy`).
