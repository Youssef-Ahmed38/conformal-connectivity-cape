# Count live GPU jobs (train/evaluate/eval_/tta_ scripts, excluding CPU runs and worker processes).
# -Name X  -> count only jobs writing to runs/X
param([string]$Name = "")
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match '(train|evaluate|eval_|tta_)\w*\.py' -and
    $_.CommandLine -notmatch 'multiprocessing' -and
    $_.CommandLine -notmatch '--device cpu'
} | Where-Object {
    $p = Get-Process -Id $_.ProcessId -ErrorAction SilentlyContinue
    $p -and -not $p.HasExited
}
if ($Name) {
    $procs = $procs | Where-Object { $_.CommandLine -match ("runs/" + [regex]::Escape($Name) + "(\s|$)") }
}
@($procs).Count
