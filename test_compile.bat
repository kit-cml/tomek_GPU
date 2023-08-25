@echo off
mingw32-make.exe clean all
REM This conditional is for checking whether the command has error or not
IF %ERRORLEVEL% NEQ 0 (
	echo Failed to compile the code!
	exit /b %ERRORLEVEL%
) 
ELSE (echo Successfully compile the code!)

cd bin
drug_sim.exe %1
IF %ERRORLEVEL% NEQ 0 (echo Something wrong with the process!) ELSE (echo Process successfuly done!)
cd ..