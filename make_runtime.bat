@echo off
docker login && lithops runtime build ismaelca/msg_python37:0.1 && echo "Fet! Recorda fer el push al repositori des de l'aplicació Docker."
pause>nul