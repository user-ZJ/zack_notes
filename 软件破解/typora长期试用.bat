@echo off
echo ������������� Typora ����
pause
echo ������ִ�����ϢֻҪ����Ȩ�޲����Ĵ���������������������
echo.

taskkill /im typora.exe /f

:: /f ����ʾ��ǿ��ɾ��
:: /v IDate ָ��Ҫɾ������
REG DELETE "HKEY_CURRENT_USER\Software\Typora" /f /v IDate
REG DELETE "HKEY_CURRENT_USER\Software\Typora" /f /v SLicense

:: version 0.x & version 1.x
del /F /A "%APPDATA%\Typora\profile.data"
del /F /A "%APPDATA%\Typora\history.data"

:: version 0.x
del /F /A "%APPDATA%\Typora\Cookies"

:: version 1.x
del /F /A "%APPDATA%\Typora\Local State"

:: д�� {"didShowWelcomePanel2":true}
echo 7b2264696453686f7757656c636f6d6550616e656c32223a747275657d > "%APPDATA%\Typora\profile.data"
echo.
echo ���

:: �ȴ���һ��������
:: pause

:: ���� Typora
start typora