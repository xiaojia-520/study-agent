@echo off
title 模型自动下载（仅使用 git，已存在则跳过）
chcp 65001 >nul

set "ROOT=%~dp0"
set "ASR_DIR=%ROOT%data\models\asr"
set "PUNC_DIR=%ROOT%data\models\punc"
set "EMBED_DIR=%ROOT%data\models\embedding"

set "ASR_URL=https://www.modelscope.cn/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch.git"
set "PUNC_URL=https://www.modelscope.cn/iic/punc_ct-transformer_cn-en-common-vocab471067-large.git"
set "EMBED_URL=https://www.modelscope.cn/Xorbits/bge-small-zh-v1.5.git"

where git >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 git，请先安装 git。
    pause
    exit /b
)

call :download "%ASR_URL%" "%ASR_DIR%"
call :download "%PUNC_URL%" "%PUNC_DIR%"
call :download "%EMBED_URL%" "%EMBED_DIR%"

echo.
echo ✅ 所有模型处理完成。
pause
exit /b

:: -------------------------
:: 子程序：下载（已存在则跳过）
:: -------------------------
:download
set "URL=%~1"
set "DIR=%~2"

:: 情况1：已是 Git 仓库 → 视为已存在
if exist "%DIR%\.git" (
    echo ✅ 模型已存在：%DIR% （Git 仓库）— 跳过
    echo.
    exit /b 0
)

:: 情况2：目录存在且非空（不是 Git 仓库）→ 视为已存在
if exist "%DIR%" (
    dir /b "%DIR%" >nul 2>nul
    if not errorlevel 1 (
        echo ✅ 模型已存在：%DIR% （非空目录）— 跳过
        echo.
        exit /b 0
    )
)

:: 保证父目录存在
if not exist "%DIR%" (
    mkdir "%DIR%"
) else (
    :: 目录存在但为空；为避免 git clone 目标已存在的意外，先清空再克隆
    for /f %%A in ('dir /b "%DIR%" 2^>nul') do set _has_any=1
    if not defined _has_any (
        rmdir /s /q "%DIR%"
    )
    set "_has_any="
)

echo 正在下载模型：%URL%
git clone --depth 1 "%URL%" "%DIR%"
if errorlevel 1 (
    echo ❌ 下载失败：%URL%
) else (
    echo ✅ 下载完成：%DIR%
)
echo.
exit /b 0
