@echo off
where java >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Java was not found. Install JDK 17 and set JAVA_HOME or PATH.
    pause
    exit /b 1
)

java -version
python -m yan_gua %*
if errorlevel 1 echo [ERROR] YanGua exited with an error.
pause
