@ECHO OFF
set repo=https://github.com/unibz-tessaris/stem2025-rl.git
FOR /F "tokens=* USEBACKQ" %%F IN (`powershell -C "[Environment]::GetFolderPath([Environment+SpecialFolder]::Desktop)"`) DO (
  SET desk=%%F
)
set dest=%desk%\stem2025-rl

ECHO %repo%
ECHO %dest%

pixi exec git config --global http.schannelCheckRevoke false
pixi exec git clone "%repo%" "%dest%"

cd "%dest%"
pixi exec git fetch
pixi exec git reset --hard "@{u}"
pixi exec git clean -xdf -e .pixi/

pixi run lab

ECHO done
