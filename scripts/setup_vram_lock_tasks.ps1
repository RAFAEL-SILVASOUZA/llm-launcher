# Setup unico (rodar ELEVADO) para o toggle de VRAM lock do app.
#
# Registra duas tarefas agendadas on-demand que rodam como SYSTEM:
#   VRAMLockOn  -> nvidia-smi --lock-memory-clocks=10251
#   VRAMLockOff -> nvidia-smi --reset-memory-clocks
# e libera Authenticated Users para DISPARA-las (GRGX no SDDL). Assim o
# backend do launcher (nao-elevado) consegue ligar/desligar o lock via
# `schtasks /Run` sem UAC.
#
# Nao mexe na tarefa de boot "Lock VRAM Clock 3090Ti" nem no
# C:\Users\PC\lock_vram_clock.ps1 -- apos reboot o lock volta sozinho.
# Idempotente: re-rodar apenas re-registra as tarefas.

$ErrorActionPreference = "Stop"

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
                 -ExecutionTimeLimit (New-TimeSpan -Minutes 2)
$smi = "C:\Windows\System32\nvidia-smi.exe"

$tasks = @(
    @{ Name = "VRAMLockOn";  Args = "--lock-memory-clocks=10251" },
    @{ Name = "VRAMLockOff"; Args = "--reset-memory-clocks" }
)

foreach ($t in $tasks) {
    $action = New-ScheduledTaskAction -Execute $smi -Argument $t.Args
    Register-ScheduledTask -TaskName $t.Name -Action $action -Principal $principal `
        -Settings $settings -Force | Out-Null
    Write-Host "Registrada: $($t.Name)"
}

# BA/SY full control; AU (Authenticated Users) read+execute = pode iniciar a tarefa.
$sddl = "D:(A;;FA;;;BA)(A;;FA;;;SY)(A;;GRGX;;;AU)"
$svc = New-Object -ComObject Schedule.Service
$svc.Connect()
$folder = $svc.GetFolder("\")
foreach ($t in $tasks) {
    $folder.GetTask($t.Name).SetSecurityDescriptor($sddl, 0)
    Write-Host "SDDL aplicado: $($t.Name)"
}

Write-Host "OK - toggle de VRAM lock pronto para o app."
