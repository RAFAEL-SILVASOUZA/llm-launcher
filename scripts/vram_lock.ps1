# Trava o clock de memória de cada GPU no MÁXIMO DELA.
#
# Por que existe: em carga CUDA o driver da NVIDIA rebaixa o clock da VRAM (Ampere cai
# pra P2/P3). Não é throttle térmico nem de power — a placa nem chega perto do limite de
# energia. Decode de LLM é memory-bandwidth bound, então isso custa throughput direto.
# Medido 2026-07-14 (Qwen3.6-27B, dual GPU): 3090 preso em P3/5001 MHz dava 18,5-19,1 t/s;
# travado em 9501 MHz, 24,6-24,9 t/s. +30%.
#
# Por que NÃO usa um valor fixo: a tarefa antiga (VRAMLockOn) rodava
# `--lock-memory-clocks=10251` sem -i, valor válido só pro 3090 Ti. Com o 3090 no rig
# (máx 9751) isso é clock inválido. Aqui cada GPU é travada no máximo que ELA suporta.
#
# CUSTO: com o clock travado a VRAM não desce em idle — a placa fica mais quente e puxa
# mais energia parada (~27W -> ~91W medido em 2026-06). Foi por isso que a versão anterior
# desta feature foi removida do app. Rodar com -Reset desfaz.
#
# Uso (exige PowerShell ELEVADO):
#   .\vram_lock.ps1           # trava todas as GPUs no clock máximo de cada uma
#   .\vram_lock.ps1 -Reset    # devolve o controle ao driver

param([switch]$Reset)

$smi = "$env:SystemRoot\System32\nvidia-smi.exe"
if (-not (Test-Path $smi)) { Write-Error "nvidia-smi não encontrado em $smi"; exit 1 }

# Falha cedo e com mensagem clara: sem elevação o nvidia-smi recusa a mudança de clock.
$admin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) {
    Write-Error "Precisa de PowerShell como Administrador (nvidia-smi recusa alterar clocks sem elevação)."
    exit 1
}

$gpus = & $smi --query-gpu=index,name,clocks.max.memory --format=csv,noheader,nounits
foreach ($line in $gpus) {
    $p    = $line -split ',\s*'
    $idx  = $p[0].Trim()
    $name = $p[1].Trim()
    $max  = $p[2].Trim()

    if ($Reset) {
        & $smi -i $idx -rmc | Out-Null
        Write-Host "GPU $idx ($name): clock de memória devolvido ao driver"
    } else {
        # O driver arredonda pro clock suportado mais próximo (pedir 9751 no 3090 dá 9501).
        & $smi -i $idx -lmc $max | Out-Null
        $now = (& $smi -i $idx --query-gpu=clocks.current.memory --format=csv,noheader,nounits).Trim()
        Write-Host "GPU $idx ($name): travada em $now MHz (pedido: $max)"
    }
}
