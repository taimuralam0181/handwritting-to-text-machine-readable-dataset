Set shell = CreateObject("WScript.Shell")
projectPath = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)
shell.CurrentDirectory = projectPath
shell.Run """" & projectPath & "\RUN_APP.bat" & """", 0, False
