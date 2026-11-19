# Adds "TBOT Console" to the Desktop and the Start menu (run once; run again after moving the repo).
#   powershell -ExecutionPolicy Bypass -File software\desktop\install_shortcut.ps1
# Needs Python 3.10+ for Windows (python.org); no packages, no internet.
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$app = Join-Path $here "tbot_console.pyw"
$ico = Join-Path $here "tbot.ico"

$pyw = $null
foreach ($name in "pythonw.exe", "pyw.exe") {
    $c = Get-Command $name -ErrorAction SilentlyContinue | Where-Object { $_.Source -notlike "*WindowsApps*" } | Select-Object -First 1
    if ($c) { $pyw = $c.Source; break }
}
if (-not $pyw) { throw "Python for Windows was not found (install it from python.org, with 'Add to PATH')." }

$shell = New-Object -ComObject WScript.Shell
$targets = @(
    (Join-Path ([Environment]::GetFolderPath("Desktop")) "TBOT Console.lnk"),
    (Join-Path ([Environment]::GetFolderPath("Programs")) "TBOT Console.lnk")
)
foreach ($lnk in $targets) {
    $s = $shell.CreateShortcut($lnk)
    $s.TargetPath = $pyw
    $s.Arguments = "`"$app`""
    $s.WorkingDirectory = $here
    $s.IconLocation = "$ico,0"
    $s.Description = "TBOT Console: set up, calibrate and review the gun"
    $s.Save()
    Write-Host "created $lnk"
}
Write-Host "Plug the gun's Ethernet cable into this computer, power it on and open TBOT Console."
