# Monitor COM3 for STM32 N657 board output
# Usage: .\monitor_com3.ps1

param(
    [string]$PortName = "COM3",
    [int]$BaudRate = 115200
)

Write-Host "Monitoring $PortName at $BaudRate baud..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

try {
    # Load the SerialPort class
    Add-Type -AssemblyName System.IO.Ports
    
    # Create and configure the serial port
    $port = New-Object System.IO.Ports.SerialPort($PortName, $BaudRate, "None", 8, "One")
    $port.ReadTimeout = 500
    
    # Open the port
    $port.Open()
    Write-Host "Connected to $PortName. Waiting for data..."
    
    # Read data continuously
    while ($port.IsOpen) {
        if ($port.BytesToRead -gt 0) {
            $data = $port.ReadExisting()
            Write-Host -NoNewline $data
        }
        Start-Sleep -Milliseconds 100
    }
}
catch {
    Write-Error "Error: $_"
}
finally {
    if ($port -and $port.IsOpen) {
        $port.Close()
        Write-Host "Port closed."
    }
}