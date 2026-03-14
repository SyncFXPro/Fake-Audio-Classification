# Deduplication: delete PNGs in results/ that were saved at 11 AM; keep 16:00 (newer model).
# Run manually from project root: .\dedupe-results-11am.ps1
# Optional -WhatIf: show what would be deleted without deleting.

param([switch]$WhatIf)

$dirs = @(
    (Join-Path $PSScriptRoot "results"),
    (Join-Path $PSScriptRoot "backend\results")
)

$deleted = 0
foreach ($dir in $dirs) {
    if (-not (Test-Path $dir)) { continue }
    Get-ChildItem -Path $dir -Filter "*.png" -File | Where-Object { $_.LastWriteTime.Hour -eq 11 } | ForEach-Object {
        Write-Host ("Would delete (11 AM): " + $_.FullName)
        if (-not $WhatIf) {
            Remove-Item $_.FullName -Force
            $deleted++
        }
    }
}

if ($WhatIf) {
    Write-Host "Run without -WhatIf to actually delete."
} else {
    Write-Host ("Deleted " + $deleted + " file(s) from 11 AM. 16:00 files kept.")
}
