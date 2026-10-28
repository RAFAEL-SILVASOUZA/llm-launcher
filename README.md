# <img src="app/public/logo.png" width="42" alt="LLM Launcher logo" align="top"> LLM Launcher

![Plataforma](https://img.shields.io/badge/plataforma-Windows%2010%2F11-0078D6.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)
![Node.js](https://img.shields.io/badge/Node.js-18%2B-339933.svg?logo=nodedotjs&logoColor=white)
![Electron](https://img.shields.io/badge/Electron-React%20%2B%20FastAPI-47848F.svg?logo=electron&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg?logo=nvidia&logoColor=white)
![llama.cpp](https://img.shields.io/badge/motor-llama.cpp-orange.svg)
![Licença](https://img.shields.io/badge/licen%C3%A7a-PolyForm%20Noncommercial%201.0.0-blue.svg)

Launcher visual + CLI para rodar modelos GGUF locais com **llama.cpp** no Windows.

O projeto tem duas frentes que compartilham as mesmas configurações:

| Frente | O que é | Quando usar |
|---|---|---|
| **APP** (`app/`) | Aplicativo desktop (Electron + React + FastAPI) com grid de configurações, estimador de VRAM, download do HuggingFace, telemetria NVIDIA e gerenciador de servidores MCP | Uso do dia a dia, tudo visual |
| **CLI** (`models.py`) | Script interativo de terminal que monta e executa a linha de comando do `llama-server`/`llama-cli`, com auto-degrade quando o servidor cai | Terminal, automação, máquinas sem interface |

![Tela inicial](telas/tela%20inicial.png)

---

## Sumário

1. [Requisitos](#1-requisitos)
2. [Instalação do zero](#2-instalação-do-zero)
   - [2.1 Clonar este repositório](#21-clonar-este-repositório)
   - [2.2 Clonar e compilar o llama.cpp](#22-clonar-e-compilar-o-llamacpp)
   - [2.3 Instalar as dependências do Python](#23-instalar-as-dependências-do-python)
   - [2.4 Configurar os caminhos](#24-configurar-os-caminhos)
3. [A APP](#3-a-app)
   - [3.1 Rodando em desenvolvimento](#31-rodando-em-desenvolvimento)
   - [3.2 Gerando o executável](#32-gerando-o-executável)
   - [3.3 Arquitetura](#33-arquitetura)
   - [3.4 As telas](#34-as-telas)
4. [O CLI (models.py)](#4-o-cli-modelspy)
5. [Onde ficam os arquivos de configuração](#5-onde-ficam-os-arquivos-de-configuração)
6. [Auto-degrade: o que acontece quando o servidor cai](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai)
7. [Solução de problemas](#7-solução-de-problemas)
8. [Licença](#8-licença)

---

## 1. Requisitos

Antes de começar, instale (tudo gratuito):

| Ferramenta | Para quê | Onde baixar |
|---|---|---|
| **Git** | clonar os repositórios | https://git-scm.com/download/win |
| **Python 3.10+** | backend da APP e o CLI | https://www.python.org/downloads/ (marque **"Add python.exe to PATH"** no instalador) |
| **Node.js 18+ (LTS)** | frontend/Electron da APP | https://nodejs.org/ |
| **Visual Studio 2022 Community** | compilar o llama.cpp | https://visualstudio.microsoft.com/ — no instalador, selecione a carga de trabalho **"Desenvolvimento para desktop com C++"** |
| **CMake 3.24+** | compilar o llama.cpp | já vem com o Visual Studio; ou https://cmake.org/download/ |
| **CUDA Toolkit 12.x** | aceleração por GPU NVIDIA | https://developer.nvidia.com/cuda-downloads (instale **depois** do Visual Studio) |

> 💡 GPU NVIDIA é fortemente recomendada. Sem GPU o llama.cpp roda só em CPU — funciona, porém muito mais lento (nesse caso pule o CUDA Toolkit e compile sem `-DGGML_CUDA=ON`).

---

## 2. Instalação do zero

### 2.1 Clonar este repositório

Abra o **PowerShell** e rode:

```powershell
git clone https://github.com/RAFAEL-SILVASOUZA/llm-launcher.git C:\Users\SEU_USUARIO\llm
cd C:\Users\SEU_USUARIO\llm
```

### 2.2 Clonar e compilar o llama.cpp

O launcher **não traz o llama.cpp junto** — ele executa os binários `llama-server.exe` / `llama-cli.exe` que você compila. Ele conhece quatro *backends* (builds diferentes):

| Backend | O que é | Obrigatório? |
|---|---|---|
| `vanilla` | llama.cpp oficial (upstream), com os parsers de chat/tool mais novos e MTP nativo | ✅ **Sim — comece por ele** |
| `mtp` | Não é um build separado: é o mesmo binário do `vanilla` lançado com `--spec-type draft-mtp` (decodificação especulativa para modelos com head MTP) | Vem de graça com o vanilla |
| `turbo` | Fork customizado ("turboquant") com tipos extras de KV cache `turbo2/3/4` (compressão agressiva para contexto longo) | Opcional — se não tiver, use os KV padrão (`f16`, `q8_0`…) |
| `custom` | **Qualquer outro build** — o fork que um modelo específico exige (arch nova ainda não mergeada, patch de tokenizer…). Não tem caminho fixo: você aponta a pasta dos binários em **Settings → Diretórios dos binários llama.cpp** (ou no menu Settings do CLI) e troca quando quiser experimentar outro | Opcional |

Sobre o `custom`: nada é assumido sobre o build. Os tipos de KV cache oferecidos saem do `--help` do binário que você apontou (então um fork com KV próprio aparece com os tipos dele), e o *speculative MTP* fica desligado — emitir `--spec-type draft-mtp` num fork desconhecido quebra o load e o auto-degrade não sabe remover essa flag. Se o build for exigente quanto a flags, combine `custom` com o checkbox **"llama.cpp decide (comando mínimo)"**.

**Passo a passo (build vanilla com CUDA):**

```powershell
# 1. Clone o llama.cpp oficial
git clone https://github.com/ggml-org/llama.cpp C:\Projetos\llama-cpp
cd C:\Projetos\llama-cpp

# 2. Configure o build (com suporte a GPU NVIDIA)
cmake -B build -DGGML_CUDA=ON

# 3. Compile em modo Release (demora ~10-30 min na primeira vez)
cmake --build build --config Release -j
```

Ao final, os binários estarão em:

```
C:\Projetos\llama-cpp\build\bin\Release\llama-server.exe
C:\Projetos\llama-cpp\build\bin\Release\llama-cli.exe
```

> **Sem GPU NVIDIA?** Troque o passo 2 por `cmake -B build` (sem `-DGGML_CUDA=ON`).
>
> **Erro de CUDA não encontrado?** Confirme que o CUDA Toolkit foi instalado *depois* do Visual Studio e reabra o PowerShell.

Para **atualizar** o llama.cpp no futuro:

```powershell
cd C:\Projetos\llama-cpp
git pull
cmake --build build --config Release -j
```

Se você tiver o fork turboquant, o processo é idêntico — só muda a URL do clone e a pasta (ex.: `C:\Projetos\llama-cpp-turboquant`).

### 2.3 Instalar as dependências do Python

```powershell
# Backend da APP (FastAPI + uvicorn)
pip install -r app\requirements.txt

# CLI interativo
pip install questionary
```

### 2.4 Configurar os caminhos

O launcher precisa saber **duas coisas**: onde estão seus **modelos (.gguf)** e onde estão os **binários do llama.cpp**.

Existem dois lugares onde isso é definido:

1. **Defaults no código** — `app/api/core/constants.py` (APP) e o topo de `models.py` (CLI) trazem caminhos padrão:
   - Modelos: `D:\models` (pasta do LM Studio)
   - Binários vanilla: `C:\Projetos\llama-cpp\build\bin\Release`
   - Binários turbo: `C:\Projetos\llama-cpp-turboquant\build\bin\Release`

   Se seus caminhos forem outros, ajuste essas constantes **ou** use a opção 2 (recomendado):

2. **Tela Settings da APP** (ou modo `settings` do CLI) — grava um `app_settings.json` compartilhado entre APP e CLI, com:
   - `model_paths`: lista de pastas varridas em busca de `.gguf`;
   - `backend_paths`: pasta dos binários de cada backend (sobrescreve o default sem tocar no código). É também como o backend `custom` ganha um caminho — ele não tem default utilizável.

   ![Tela Settings](telas/tela%20de%20configurações.png)

   O que foi configurado na APP vale para o CLI e vice-versa — é o mesmo arquivo, inclusive para os `backend_paths`.

> 📁 **Organização dos modelos:** o scanner procura `.gguf` recursivamente dentro das pastas cadastradas. A convenção `dono/repositorio/quant/arquivo.gguf` (a mesma do LM Studio e do download interno da APP) deixa a listagem mais legível.

---

## 3. A APP

### 3.1 Rodando em desenvolvimento

```powershell
cd app
npm install        # primeira vez
npm run dev        # abre Vite + Electron; o Electron sobe o backend Python sozinho
```

O Electron inicia o backend FastAPI automaticamente (procura `python` no PATH; para usar outro interpretador, defina a variável de ambiente `PYTHON`).

### 3.2 Gerando o executável

```powershell
cd app
npm run build
```

Sai um executável **portable** para Windows (via electron-builder) na pasta `app/release`. O Python continua sendo necessário na máquina — o backend roda como processo filho. Se a APP abrir com erro de conexão, o caminho do log do backend (`python.log`) aparece na própria tela para diagnóstico.

**Releases no GitHub:** o repositório tem um workflow de CI (`.github/workflows/release.yml`) que faz tudo isso sozinho. Basta criar e enviar uma tag de versão:

```powershell
git tag v0.2.0
git push origin v0.2.0
```

O GitHub Actions builda a APP num runner Windows, compacta o portable em `LLM-Launcher-v0.2.0-win-portable.zip` e publica uma **Release** com o zip anexado e as notas geradas a partir dos commits. A versão do `package.json` é sincronizada com a tag automaticamente durante o build.

### 3.3 Arquitetura

```
app/
├── electron/     # main.js: cria a janela e sobe o backend Python
├── src/          # frontend React + Tailwind (Vite)
│   └── components/   # uma página por aba + modais
└── api/          # backend FastAPI
    └── core/     # builder do comando, runner com auto-degrade, estimador
                  # de VRAM, scanner de GGUF, download HF, nvidia-smi, MCP
```

O frontend conversa com o backend por REST + SSE (eventos de launch e de download em tempo real). O backend é quem executa o `llama-server` — se a janela for fechada com um servidor rodando, ao reabrir a APP ela **reanexa** o processo órfão e volta a oferecer o botão de Stop.

### 3.4 As telas

#### Cabeçalho (presente em todas as telas)

Na faixa superior ficam as barras de **VRAM** e **RAM** em tempo real (atualizam a cada 2 s) e os *badges* dos backends — verde quando o binário foi encontrado no caminho configurado. Ao lado, o botão **refresh** recarrega tudo.

#### Aba Configs — a tela inicial

![Tela inicial — grid de configs](telas/tela%20inicial.png)

É o coração da APP: um grid com as configurações de launch salvas. Cada linha é um par **modelo + backend + parâmetros**:

- **Bolinha de status** (coluna `•`): estimativa de memória da config — 🟢 cabe na VRAM com folga, 🟡 apertado, 🔴 não cabe. Calculada em lote ao abrir a tela.
- **Colunas**: backend, contexto (CTX), tipo de KV cache, camadas na GPU (`-ngl`), camadas MoE na CPU (`-ncmoe`), slots paralelos (`-np`), estado do thinking/reasoning e flag de visão (mmproj).
- **Filtro e chips** no topo filtram por texto ou por backend (`todos / turbo / vanilla / mtp / custom`).
- **Ações por linha**: ✏️ editar, ⧉ duplicar, 🚀 **launch**, 🗑️ excluir.
- **Checkboxes à esquerda + launch múltiplo (modo router)**: marque várias configs e suba todas de uma vez — o `llama-server` entra em modo router na porta única (1234) e o cliente escolhe o modelo pelo campo `model` da requisição. Só é permitido **um launch ativo por vez** (tentar outro retorna erro 409).
- **+ nova** abre o editor com uma config em branco.

Enquanto houver um servidor no ar, a tela mostra os controles de **Stop**, **Restart** e **abrir logs**.

#### Editor de configuração (modal "Editar configuração")

![Setup do modelo — parte 1](telas/tela%20de%20setup%20do%20modelo%20-%2001.png)

Abre ao criar/editar/duplicar uma config. À esquerda, os parâmetros; à direita, dois painéis que **atualizam ao vivo** conforme você mexe:

- **Estimativa**: barras de VRAM/RAM previstas para essa config (pesos + KV cache + compute), com *breakdown* expansível e metadados do GGUF (camadas, embedding, heads, arquitetura). Considera a VRAM já ocupada pelo sistema.
- **Comando**: a linha de comando exata do `llama-server` que será executada — dá para copiar e rodar manualmente se quiser.

Campos da primeira parte:

- **Modelo (.gguf)** — dropdown com tudo que o scanner achou nas pastas configuradas.
- **llama.cpp decide (comando mínimo)** — checkbox ao lado do dropdown. Marcado, o comando sai só com o modelo, o `--mmproj` (se houver), o `--alias` e o servidor HTTP (`--jinja --metrics --host --port`); **nenhuma** flag de tuning é emitida e o llama.cpp escolhe contexto, `-ngl`, KV, threads, samplers e reasoning sozinho. Os demais campos continuam salvos na config (aparecem esmaecidos) e voltam a valer assim que você desmarcar. Vale para o modo router também — a seção do preset fica só com o modelo. Em compensação, o [auto-degrade](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai) fica desligado: sem flags nossas no comando, não há nada para baixar.
- **Backend** — turboquant / vanilla / mtp (cards com descrição; desabilitado se o binário não existir).
- **Context window** e **Slots paralelos (`-np`)** — o launcher emite `--kv-unified`, então o contexto é um **pool único** que os slots dividem (é o que o LM Studio faz por padrão): o KV custa `contexto × 1`, e não `× slots`. Uma request sozinha pode usar o pool inteiro; várias simultâneas disputam. Se o build não tiver a flag, o `-c` volta a ser multiplicado pelos slots para não encolher a janela em silêncio — e aí o KV custa `× slots` mesmo.
- **KV cache (`--cache-type-k/v`)** — de `f16` (padrão) a quantizados (`q8_0`, `q4_0`…); os tipos `turbo2/3/4` só aparecem no backend turboquant.
- **Flash Attention (`-fa`)** — liga/desliga.
- **GPU / Offload (`-ngl`)** — quantas camadas vão para a GPU, com botão **"sugerir -ngl"** que calcula o máximo que cabe na VRAM. Para modelos MoE há também o `-ncmoe` (camadas de experts na CPU).
- **Avisos inteligentes** — ex.: ao detectar arquitetura híbrida SSM, sugere desligar o reasoning (evita travamento em loops agênticos); com 2 GPUs, expõe **split mode (`-sm`)** e **proporção (`-ts`)**.

![Setup do modelo — parte 2](telas/tela%20de%20setup%20do%20modelo%20-%2002.png)

Rolando o modal, a segunda parte:

- **Server (KV total, prompt cache, checkpoints)** — `cache-ram` (MiB de RAM para cache de prompts; acelera reuso de contexto) e `ctx-checkpoints` (restauração parcial de contexto editado).
- **Generation & misc** — `max-tokens (-n)` (proteção contra loop infinito), `batch (-b)`, `ubatch (-ub)` (pico de VRAM no prefill), threads de geração e de batch (`-t` / `-tb`).
- **Flags**: `--mlock`, `--verbose`, e o modo de execução — **server (HTTP)** ou **cli** (chat no terminal).

Botões: **Salvar** (só grava) ou **Salvar e launch** (grava e sobe o servidor imediatamente).

#### Modal de logs do launch

![Modal de logs](telas/modal%20de%20logs.png)

Abre automaticamente ao lançar. Mostra:

- Resumo da config no topo (backend, ctx, kv, ngl, np);
- O **comando executado** e o log do `llama-server` em tempo real (via SSE);
- Cada **tentativa** numerada — se o servidor cair na carga, o auto-degrade ajusta a config e tenta de novo (ver [seção 6](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai));
- Botões **Reiniciar** (mata e ressobe com a mesma config, mantendo a porta — o cliente reconecta sozinho), **Cancelar** (derruba o servidor) e **Esconder** (fecha o modal sem parar nada; reabra pelo botão de logs na aba Configs).

Quando a carga termina com sucesso, o servidor fica ouvindo em `http://localhost:1234` com API compatível com OpenAI — aponte Claude Code, Continue, ou qualquer cliente OpenAI-compatible para ele.

#### Aba Models

![Tela de modelos](telas/tela%20de%20modelos.png)

Inventário de tudo que o scanner encontrou nas pastas configuradas:

- **Alias + caminho relativo**, **tamanho em disco** (somando shards de modelos divididos);
- **Flags** detectadas pelos metadados do GGUF: 🧠 thinking/reasoning, ⚡ MTP, 🖼️ visão (mmproj);
- **delete** remove o `.gguf` do disco (e o mmproj associado), com confirmação e limpeza de pastas vazias.

A head MTP pode vir de dois jeitos, e o app trata os dois sozinho:

- **embutida** no gguf do modelo (repos tipo `Qwen3.6-27B-MTP-GGUF`) — basta o `--spec-type draft-mtp`;
- **em gguf separado**, numa pasta irmã com prefixo `mtp-` (`Tess-4-27B-GGUF/{Tess-4-27B-Q8_0, mtp-Tess-4-27B-Q8_0}`) — aí o comando ganha também `-md <caminho da head>`, sem o qual o llama.cpp montaria o contexto MTP em cima do modelo alvo, que não tem essa camada.

O drafter não aparece na lista de modelos (sozinho ele não roda: é uma camada só), pesa na estimativa de VRAM como um segundo modelo (~3 GB num 27B q8_0) e é apagado junto com o alvo no **delete**.

O campo **filtrar…** busca por nome/caminho.

#### Aba Download

![Tela de download](telas/tela%20de%20downloads.png)

Baixa modelos direto do HuggingFace:

1. **Salvar em** — escolha a pasta de destino entre as cadastradas no Settings (é obrigatório ter ao menos uma).
2. **HuggingFace download** — cole a URL (`https://huggingface.co/dono/repo`) ou o id (`dono/repo`) e clique **inspecionar**: a APP lista os arquivos GGUF do repositório agrupados por quantização para você escolher qual baixar. Ou use o campo de **busca** (ex.: `qwen 30b moe coder`) para pesquisar repositórios GGUF no HF.
3. Durante o download: barra de progresso por arquivo com velocidade, suporte a modelos multi-shard, botão de cancelar. Pode **trocar de aba sem medo** — o download continua no backend e a aba Download mostra um spinner enquanto estiver ativo.
4. Ao final, a integridade é validada e a lista de modelos recarrega sozinha.
5. A APP busca o **`generation_config.json`** do repo (ou do `base_model`, quando é repo de quant) e grava um **`sampling.json`** ao lado do modelo — são os samplers que o *autor* publicou. Se o repo não publicar nenhum, a tela avisa.

#### Samplers por modelo

Temperatura, top-p, top-k, min-p e repeat-penalty **não são um preset único** — são resolvidos por modelo, e a APP mostra **de onde vieram**:

| Procedência | Significado |
|---|---|
| `valores do autor` | vieram do `generation_config.json` do repo (via `sampling.json`) |
| `preset reasoning` | o chat template do GGUF tem `<think>` → temp 0.6 / top-p 0.95, sem penalidade |
| `preset código` | template sem `<think>` → temp 0.3 / top-p 0.8 / repeat 1.05 (validado em A/B para agente de código) |
| `chute` | o GGUF não pôde ser lido — confira o card do modelo |
| `fixado por você` | você editou na seção **Sampling** do editor; nada sobrescreve |

Por que isso importa: em modelo de raciocínio, temperatura baixa e repeat-penalty **aumentam** o loop dentro do `<think>` e punem a repetição legítima de fórmula/identificador. Quem pensa e quem não pensa precisam de presets diferentes — e a detecção sai do **chat template**, nunca do nome do arquivo (`ornith-1.0-35b` é reasoning e não tem nenhuma palavra reveladora no nome).

#### Aba MCP

![Tela de MCPs](telas/tela%20de%20MCPs.png)

Gerenciador simples de **servidores MCP locais** (Model Context Protocol, transporte HTTP stream): cadastre processos que você liga/desliga manualmente pela APP — útil para dar ferramentas (filesystem, busca…) a agentes apontados para o seu llama-server.

- Cada card mostra nome, diretório e comando, com botões **ligar/desligar**, 👁️ ver logs (stdout/stderr capturados), ✏️ editar e 🗑️ excluir.
- Se o processo cair, o status volta para desligado automaticamente.

![Cadastro de MCP](telas/tela%20MCPs%20-%20modal%20cadastro.png)

O cadastro pede só três campos: **Nome**, **Diretório (cwd)** e **Comando** (ex.: `npm run dev`), executado via shell no diretório informado.

#### Aba NVIDIA

![Tela NVIDIA](telas/tela%20NVIDIA.png)

Telemetria completa por GPU (via `nvidia-smi`), com **auto-refresh de 2 s**:

- VRAM usada/total/livre, uso de GPU e de memória, temperaturas (GPU, memória, hot spot, margem até o T-Limit), fan, potência, clocks SM e de memória, P-state e versão do driver;
- Gráficos de **histórico** (VRAM, GPU, RAM) para acompanhar o comportamento durante a inferência — útil para flagrar, por exemplo, a GPU presa em P-state baixo derrubando o clock de memória.

#### Aba Settings

![Tela Settings](telas/tela%20de%20configurações.png)

Onde os caminhos são configurados (ver [seção 2.4](#24-configurar-os-caminhos)):

- **Pastas de modelos** — adicione/remova quantas quiser; a aba Models revarre ao salvar. A primeira pasta é a sugerida como destino de download.
- **Diretórios dos binários llama.cpp** — um campo por backend (turbo, vanilla e custom; o mtp herda do vanilla). Vazio = usa o default do código; o texto `usando default: …` mostra qual é. O botão ↺ restaura o default. Informe a **pasta** que contém os `.exe`, não o executável.
- O campo **custom** é o único cujo default não existe de fábrica: é onde você aponta o build específico de um modelo. Enquanto estiver vazio, o backend `custom` aparece no editor como *"sem caminho — configure em Settings"* e não pode ser lançado. O CLI lê o mesmo arquivo (menu **settings → 🧰 Diretórios dos binários llama.cpp**), então configurar num lado vale nos dois.

---

## 4. O CLI (models.py)

Tudo que a APP faz de essencial, em modo texto interativo (setas ↑↓ + Enter):

```powershell
cd C:\Users\SEU_USUARIO\llm
python models.py
```

### Menu inicial

```
⚙️  Modo de execução:
❯ llama-server  — API HTTP (OpenAI-compatible, Claude Code, etc.)
  llama-cli     — Chat interativo no terminal
  download      — Baixar modelo do HuggingFace
  delete        — Apagar um modelo (.gguf) e o mmproj associado do disco
  settings      — Configurar pastas onde procurar modelos (.gguf)
  nvidia-status — Status completo da GPU NVIDIA
```

### Modo `llama-server` (o principal)

1. **Backend** — turbo / vanilla / mtp / custom / dflash (só aparecem os que têm binário no caminho configurado — inclusive o override de pasta feito em Settings; o CLI ainda suporta o backend experimental `dflash`, de decodificação especulativa com drafter GGUF separado, que não existe na APP).
2. **Modelo** — lista tudo que foi achado nas pastas configuradas, marcando modelos com visão (🖼️).
3. **Configuração salva?** — se você já rodou esse par modelo+backend antes, o CLI mostra a config anterior num quadro e pergunta se quer reutilizá-la. Respondendo **sim**, pula direto para o launch. As configs são as **mesmas da APP** (mesmo arquivo).
4. Caso contrário, a primeira pergunta é **"deixar o llama.cpp decidir tudo?"** — respondendo **sim**, o CLI só pergunta mmproj e verbose e sobe com o comando mínimo (equivale ao checkbox da APP; nada de tuning entra na linha de comando e o auto-degrade fica desligado).
5. Não sendo auto, pergunta passo a passo: **context window**, **KV cache** (o CLI *sonda o binário* e só oferece os tipos que aquele build aceita), **flash attention**, **`-ngl`** (com sugestão automática baseada na VRAM livre e menu de estimativa por opção), **`-ncmoe`** para MoE, **split entre GPUs** quando há mais de uma, **slots paralelos**, **cache-ram**, **ctx-checkpoints**, **reasoning budget** (para modelos thinking), **max-tokens**, **batch/ubatch**, **mlock**, **verbose**.
6. Antes de subir, imprime a **estimativa de memória** (as mesmas contas do painel da APP, com barras no terminal) e o **comando completo**. No modo auto a estimativa é omitida — quem escolhe ctx/`-ngl`/KV ali é o llama.cpp, então qualquer número nosso seria chute.
7. Executa com **launch resiliente**: monitora a saída, classifica falhas (OOM, KV rejeitado, flag desconhecida, GGUF corrompido…) e aplica o [auto-degrade](#6-auto-degrade-o-que-acontece-quando-o-servidor-cai) até o servidor estabilizar. A config que funcionou é salva para a próxima vez.

### Modo `llama-cli`

Mesmo fluxo de seleção, mas gera um chat interativo direto no terminal em vez do servidor HTTP.

### Modos utilitários

- **download** — igual à aba Download da APP: cola URL/id do HF ou busca por texto, escolhe a quantização, baixa com barra de progresso e validação.
- **delete** — apaga um `.gguf` (e mmproj associado) com confirmação.
- **settings** — adiciona/edita/remove as pastas de modelos (grava no mesmo `app_settings.json` da APP).
- **nvidia-status** — painel de telemetria da GPU no terminal (o equivalente da aba NVIDIA).

---

## 5. Onde ficam os arquivos de configuração

Todos são JSON simples, compartilhados entre APP e CLI:

| Arquivo | Conteúdo |
|---|---|
| `app_settings.json` | pastas de modelos (`model_paths`) e diretórios dos binários (`backend_paths`) |
| `last_config.json` | as configurações de launch salvas (o grid da aba Configs / as "configs salvas" do CLI) |
| `fail_history.jsonl` | histórico de falhas de launch e do que o auto-degrade fez (um JSON por linha, para diagnóstico) |

Por padrão eles moram na pasta do fork turboquant (`C:\Projetos\llama-cpp-turboquant\`); o local é definido por `CONFIG_FILE` em `app/api/core/constants.py` e em `models.py` — se você não usa o fork, ajuste essa constante para qualquer pasta sua.

---

## 6. Auto-degrade: o que acontece quando o servidor cai

Subir um modelo grande é tentativa e erro: a estimativa pode dizer que cabe e a VRAM real dizer que não. Em vez de simplesmente falhar, o runner (APP e CLI usam a mesma lógica):

1. Lê o log do `llama-server` e **classifica a falha**: OOM de VRAM, tipo de KV não suportado pelo build, flag desconhecida, erro de mmproj, GGUF corrompido/incompleto, crash genérico.
2. Aplica **um passo de degrade** adequado à falha — por exemplo: reduzir `-ngl` (menos camadas na GPU), subir `-ncmoe` (mais experts na CPU), trocar o KV cache por um tier mais comprimido, remover a flag não suportada.
3. Tenta de novo, e repete até estabilizar ou esgotar as opções.

Regras importantes:

- **A janela de contexto nunca é cortada** — o degrade sacrifica velocidade (mais coisas na CPU), não a capacidade que você pediu.
- **GGUF corrompido não entra na escada** — é erro de download (shard faltando etc.); o runner desiste na hora e avisa, porque degradar não conserta arquivo quebrado.
- **Config com "llama.cpp decide" também não** — nenhuma flag nossa vai no comando, então todo degrau geraria a mesma linha e o restart viraria loop; o runner para na primeira queda e mostra o erro.
- Cada tentativa aparece numerada no modal de logs da APP (ou no terminal, no CLI), e a config final que funcionou é a que fica salva.

---

## 7. Solução de problemas

**A APP abre com "❌ conectando ao backend Python…" / erro de conexão**
O backend FastAPI não subiu. Confira: `python --version` funciona no PowerShell? Rodou `pip install -r app\requirements.txt`? A própria tela de erro mostra o caminho do `python.log` com o motivo exato.

**Badge do backend vermelho / "Nenhum binário do llama.cpp encontrado"**
O `llama-server.exe` não está no caminho esperado. Confira se a compilação (seção 2.2) terminou sem erro e se o diretório na aba **Settings** aponta para a pasta `build\bin\Release` correta.

**A lista de modelos está vazia**
Cadastre em **Settings → Pastas de modelos** a pasta onde estão seus `.gguf` (ou baixe um pela aba Download). O scanner é recursivo.

**O download exige "pasta cadastrada"**
Por segurança, só se baixa para pastas explicitamente cadastradas no Settings. Adicione a pasta primeiro.

**O modelo carrega e cai na hora (out of memory)**
Deixe o auto-degrade trabalhar — ele reduz `-ngl`/aumenta `-ncmoe` sozinho. Para evitar de vez: use o botão **sugerir -ngl** no editor, um KV cache quantizado (`q8_0`), ou uma quantização menor do modelo.

**Erro "data is not within the file bounds" / "invalid magic number"**
GGUF corrompido ou download incompleto (típico de modelo multi-shard interrompido). Apague pela aba Models e baixe de novo.

**Tokens por segundo muito baixos com a GPU ociosa**
Veja a aba **NVIDIA**: se o clock de memória estiver baixo (P-state P3 em vez de P2 durante a inferência), é o driver rebaixando a VRAM. Também confira se `-ngl` não caiu demais após um auto-degrade antigo salvo na config.

**Duas GPUs, desempenho pior que uma**
Use split mode **layer** (default), não **row** — no editor, campo "Split entre GPUs (-sm)".

---

## 8. Licença

Este projeto é distribuído sob a licença **[PolyForm Noncommercial 1.0.0](LICENSE)**. Em resumo:

- ✅ **Pode**: clonar, usar, estudar, modificar e redistribuir (inclusive com suas alterações) para qualquer fim **não comercial** — uso pessoal, estudo, pesquisa, hobby, instituições de ensino e organizações sem fins lucrativos;
- ❌ **Não pode**: vender o software, cobrar por ele ou usá-lo como parte de um produto/serviço comercial;
- 📄 Redistribuições devem manter uma cópia da licença (ou o link para ela).

O texto completo (que é o que vale juridicamente) está no arquivo [`LICENSE`](LICENSE).
