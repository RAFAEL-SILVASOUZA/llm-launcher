param(
    [int[]]$Sizes = @(4000, 16000, 32000, 48000, 64000, 80000),
    [switch]$NoPenalties   # passa B: desliga DRY/presence/repeat pra isolar causa
)
$ErrorActionPreference = "Stop"
$uri = "http://localhost:1234/v1/chat/completions"

# gera texto variado (nao-repetitivo) com ~N tokens; ~1.3 tok/palavra no Qwen
$bank = @("system","memory","network","protocol","engineer","compile","thread","kernel","module","signal",
    "router","packet","storage","driver","schedule","process","buffer","cache","vector","matrix",
    "quantum","neural","transform","gradient","tensor","encode","decode","stream","branch","merge",
    "deploy","cluster","virtual","container","monitor","balance","request","response","query","index",
    "garden","river","mountain","harvest","window","journey","pattern","silence","thunder","horizon")
$rng = New-Object System.Random(1234)
function New-Filler([int]$tokens) {
    $words = [int]($tokens / 1.3)
    $sb = New-Object System.Text.StringBuilder
    for ($i = 0; $i -lt $words; $i++) {
        [void]$sb.Append($bank[$rng.Next($bank.Count)])
        if ($i % 12 -eq 11) { [void]$sb.Append(". ") } else { [void]$sb.Append(" ") }
    }
    $sb.ToString()
}

$task = "IGNORE the reference document above; it is only archival filler. Your task: write an extremely long, detailed, well-structured essay about the complete history of computing, from the abacus to quantum computers. It MUST be at least 4000 words. Do NOT stop early. Do NOT summarize. Keep writing until you have covered every era in depth."

$label = if ($NoPenalties) { "SEM penalties (dry=0, presence=0, repeat=1.0)" } else { "config REAL do server (dry 0.8 ctx inteiro, presence 0.5, repeat 1.1)" }
Write-Host "=== TESTE EOS PRECOCE x TAMANHO DE CONTEXTO - $label ==="
Write-Host ("{0,9} {1,9} {2,11} {3,8} {4,14}" -f "ctx alvo","prompt_n","gerados","teto?","finish_reason")

foreach ($sz in $Sizes) {
    $filler = New-Filler $sz
    $body = @{
        messages = @(
            @{ role = "system"; content = "You are a helpful assistant." },
            @{ role = "user"; content = "REFERENCE DOCUMENT:`n$filler`n`n---`n$task" }
        )
        max_tokens   = 4096
        cache_prompt = $false
        seed         = 99
    }
    if ($NoPenalties) {
        $body.dry_multiplier = 0.0
        $body.presence_penalty = 0.0
        $body.repeat_penalty = 1.0
    }
    $json = $body | ConvertTo-Json -Depth 5 -Compress
    $r = Invoke-RestMethod -Uri $uri -Method Post -Body $json -ContentType "application/json" -TimeoutSec 1800
    $fin = $r.choices[0].finish_reason
    $gen = [int]$r.usage.completion_tokens
    $pn  = [int]$r.usage.prompt_tokens
    $teto = if ($fin -eq "length") { "SIM" } else { "nao" }
    Write-Host ("{0,9} {1,9} {2,11} {3,8} {4,14}" -f $sz, $pn, $gen, $teto, $fin)
}
Write-Host ""
Write-Host "Interpretacao: finish_reason=stop com poucos tokens = EOS precoce (modelo desistiu sozinho)."
Write-Host "finish_reason=length = bateu o teto de 4096 (comportamento saudavel nesta tarefa)."
