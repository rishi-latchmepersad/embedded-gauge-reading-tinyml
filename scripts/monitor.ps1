$port='COM3'
$sp = New-Object System.IO.Ports.SerialPort $port,115200,'None',8,'One'
$sp.ReadTimeout = 1000
$sp.NewLine = "`n"
try {
    $sp.Open()
    Write-Host "[OPEN] $port opened at 115200"
    $deadline = (Get-Date).AddSeconds(35)
    while((Get-Date) -lt $deadline) {
        try {
            $line = $sp.ReadLine()
            if ($line -ne $null) {
                Write-Host $line
            }
        } catch [System.TimeoutException] {
        }
    }
} finally {
    if ($sp.IsOpen) {
        $sp.Close()
    }
}