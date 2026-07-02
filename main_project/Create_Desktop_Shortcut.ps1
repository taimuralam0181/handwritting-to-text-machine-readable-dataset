$projectPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Prescription Dashboard.lnk"
$targetPath = Join-Path $projectPath "Launch_App.vbs"
$iconPath = Join-Path $projectPath "assets\app_icon.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $projectPath
$shortcut.IconLocation = $iconPath
$shortcut.Description = "Prescription Data Extraction Dashboard"
$shortcut.Save()

Write-Host "Shortcut created:" $shortcutPath
