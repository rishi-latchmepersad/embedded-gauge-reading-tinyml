@echo off
echo Monitoring COM3 port...
echo Press Ctrl+C to stop
echo ----------------------
powershell -Command "$port = new-Object System.IO.Ports.SerialPort 'COM3', 115200, None, 8, One; $port.Open(); Write-Host 'Monitoring COM3...'; while ($port.IsOpen) { if ($port.BytesToRead -gt 0) { $data = $port.ReadExisting(); Write-Host -NoNewline $data } Start-Sleep -Milliseconds 100 }"