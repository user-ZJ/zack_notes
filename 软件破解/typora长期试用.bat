@echo off
echo 接下来将会结束 Typora 进程
pause
echo 下面出现错误信息只要不是权限不够的错，都属于正常现象，请勿担心
echo.

taskkill /im typora.exe /f

:: /f 不提示，强制删除
:: /v IDate 指定要删除的项
REG DELETE "HKEY_CURRENT_USER\Software\Typora" /f /v IDate
REG DELETE "HKEY_CURRENT_USER\Software\Typora" /f /v SLicense

:: version 0.x & version 1.x
del /F /A "%APPDATA%\Typora\profile.data"
del /F /A "%APPDATA%\Typora\history.data"

:: version 0.x
del /F /A "%APPDATA%\Typora\Cookies"

:: version 1.x
del /F /A "%APPDATA%\Typora\Local State"

:: 写回 {"didShowWelcomePanel2":true}
echo 7b2264696453686f7757656c636f6d6550616e656c32223a747275657d > "%APPDATA%\Typora\profile.data"
echo.
echo 完成

:: 等待按一个键继续
:: pause

:: 启动 Typora
start typora