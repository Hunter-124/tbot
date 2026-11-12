# Flash Raspberry Pi OS onto the TBOT SD card and stage the first-boot files.
# Run elevated (raw disk writes need Administrator):
#   powershell -ExecutionPolicy Bypass -File software\pi\flash.ps1 -DiskNumber 2 -Image <...>.img -Tarball <tbot-software.tar.gz>
# It refuses any disk that is not a USB disk under 128 GB that is neither the boot nor the system disk.
param(
    [Parameter(Mandatory)] [int] $DiskNumber,
    [Parameter(Mandatory)] [string] $Image,
    [string] $Tarball,
    [string] $Log = "$env:TEMP\tbot-flash.log"
)
$ErrorActionPreference = 'Stop'
function Say($m) { $l = "$(Get-Date -Format HH:mm:ss) $m"; Write-Host $l; Add-Content -Path $Log -Value $l }
Set-Content -Path $Log -Value ''
try {
    $disk = Get-Disk -Number $DiskNumber
    Say "target: disk $DiskNumber '$($disk.FriendlyName)' $($disk.BusType) $([math]::Round($disk.Size/1GB,1)) GB"
    if ($disk.BusType -ne 'USB' -or $disk.IsBoot -or $disk.IsSystem -or $disk.Size -gt 128GB) { throw "refusing disk $DiskNumber (not a small USB card)" }
    $img = Get-Item $Image
    if ($img.Length -gt $disk.Size) { throw "image larger than disk" }

    # Take the card's volumes away from Windows so nothing remounts it mid-write.
    Get-Partition -DiskNumber $DiskNumber -ErrorAction SilentlyContinue | Where-Object DriveLetter |
        ForEach-Object { Remove-PartitionAccessPath -DiskNumber $DiskNumber -PartitionNumber $_.PartitionNumber -AccessPath "$($_.DriveLetter):\" }
    Clear-Disk -Number $DiskNumber -RemoveData -RemoveOEM -Confirm:$false
    try { Set-Disk -Number $DiskNumber -IsOffline $true; Say "disk offline" } catch { Say "offline not supported, continuing on cleared disk" }

    Say "writing $($img.Name) ($([math]::Round($img.Length/1MB)) MB)"
    $src = [IO.File]::OpenRead($img.FullName)
    $dst = New-Object IO.FileStream("\\.\PhysicalDrive$DiskNumber", [IO.FileMode]::Open, [IO.FileAccess]::ReadWrite, [IO.FileShare]::ReadWrite)
    $buf = New-Object byte[] (4MB); $done = 0; $next = 0
    while (($n = $src.Read($buf, 0, $buf.Length)) -gt 0) {
        if ($n % 512) { $n += 512 - ($n % 512) }  # pad the tail to a whole sector
        $dst.Write($buf, 0, $n); $done += $n
        if ($done -ge $next) { Say ("  {0:N0} / {1:N0} MB" -f ($done/1MB), ($img.Length/1MB)); $next += 256MB }
    }
    $dst.Flush(); $dst.Close(); $src.Close()
    Say "write complete"

    # Read back the first 64 MB and compare.
    $a = [IO.File]::OpenRead($img.FullName)
    $b = New-Object IO.FileStream("\\.\PhysicalDrive$DiskNumber", [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    $x = New-Object byte[] (64MB); $y = New-Object byte[] (64MB)
    [void]$a.Read($x, 0, $x.Length); [void]$b.Read($y, 0, $y.Length); $a.Close(); $b.Close()
    $sha = [Security.Cryptography.SHA256]::Create()
    if ([BitConverter]::ToString($sha.ComputeHash($x)) -ne [BitConverter]::ToString($sha.ComputeHash($y))) { throw "verify failed: first 64 MB differ" }
    Say "verify ok (first 64 MB)"

    try { Set-Disk -Number $DiskNumber -IsOffline $false } catch {}
    Update-Disk -Number $DiskNumber
    Start-Sleep -Seconds 3
    $boot = Get-Partition -DiskNumber $DiskNumber -PartitionNumber 1
    if (-not $boot.DriveLetter) { $boot | Add-PartitionAccessPath -AssignDriveLetter; Start-Sleep -Seconds 2; $boot = Get-Partition -DiskNumber $DiskNumber -PartitionNumber 1 }
    $root = "$($boot.DriveLetter):\"
    Say "boot partition at $root"

    Copy-Item -Force (Join-Path $PSScriptRoot 'boot\user-data') (Join-Path $root 'user-data')
    Copy-Item -Force (Join-Path $PSScriptRoot 'boot\network-config') (Join-Path $root 'network-config')
    if ($Tarball) { Copy-Item -Force $Tarball (Join-Path $root 'tbot-software.tar.gz') }
    Say "first-boot files staged: $((Get-ChildItem $root -Name | Where-Object { $_ -in 'user-data','network-config','meta-data','tbot-software.tar.gz' }) -join ', ')"
    Say "DONE"
} catch {
    Say "ERROR: $_"
    exit 1
}
