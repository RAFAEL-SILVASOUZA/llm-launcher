param(
    [int]$MaxTurns = 16,
    [switch]$NoPenalties,
    [switch]$NewConfig   # simula os novos defaults: dry janela 2048, sem presence
)
$ErrorActionPreference = "Stop"
$uri = "http://localhost:1234/v1/chat/completions"

$label = if ($NoPenalties) { "SEM penalties" } elseif ($NewConfig) { "NOVA config (dry janela 2048, sem presence)" } else { "config REAL do server" }
Write-Host "=== TESTE MULTI-TURNO (saida do proprio modelo acumulando) - $label ==="
Write-Host ("{0,5} {1,10} {2,9} {3,15}" -f "turno","ctx atual","gerados","finish_reason")

$messages = New-Object System.Collections.ArrayList
[void]$messages.Add(@{ role = "system"; content = "You are a technical writer producing an extremely long book about the history of technology. Always write long, detailed chapters." })
[void]$messages.Add(@{ role = "user"; content = "Write chapter 1 of the book (very long and detailed, at least 2000 words). Do not conclude the book; there will be many more chapters." })

$earlyStreak = 0
for ($turn = 1; $turn -le $MaxTurns; $turn++) {
    $body = @{
        messages     = $messages.ToArray()
        max_tokens   = 4096
        cache_prompt = $true
        seed         = 7
    }
    if ($NoPenalties) {
        $body.dry_multiplier = 0.0
        $body.presence_penalty = 0.0
        $body.repeat_penalty = 1.0
    }
    if ($NewConfig) {
        $body.dry_multiplier = 0.8
        $body.dry_base = 1.75
        $body.dry_allowed_length = 4
        $body.dry_penalty_last_n = 2048
        $body.presence_penalty = 0.0
        $body.repeat_penalty = 1.1
        $body.repeat_last_n = 256
    }
    $json = $body | ConvertTo-Json -Depth 6 -Compress
    $bytes = [Text.Encoding]::UTF8.GetBytes($json)
    $r = Invoke-RestMethod -Uri $uri -Method Post -Body $bytes -ContentType "application/json; charset=utf-8" -TimeoutSec 1800
    $fin = $r.choices[0].finish_reason
    $gen = [int]$r.usage.completion_tokens
    $pn  = [int]$r.usage.prompt_tokens
    $txt = $r.choices[0].message.content
    Write-Host ("{0,5} {1,10} {2,9} {3,15}" -f $turn, $pn, $gen, $fin)

    if ($fin -eq "stop" -and $gen -lt 1500) { $earlyStreak++ } else { $earlyStreak = 0 }
    if ($earlyStreak -ge 3) {
        Write-Host ">>> 3 turnos seguidos de parada precoce. Encerrando o teste aqui."
        break
    }

    [void]$messages.Add(@{ role = "assistant"; content = $txt })
    [void]$messages.Add(@{ role = "user"; content = "Continue with the next chapter. Just as long and detailed (at least 2000 words). Do not conclude the book." })
}
Write-Host ""
Write-Host "Parada precoce = finish_reason stop com poucos tokens apesar do pedido de 2000+ palavras."
