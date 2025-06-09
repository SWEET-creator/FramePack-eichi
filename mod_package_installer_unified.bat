@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo == FlowDownTheRiver Python Env Setup ^& Package Installer  ==
echo == (Copies Headers/Libs ^& Installs All Required Packages) ==
echo ============================================================
echo.
echo ** PREREQUISITE **
echo This script expects 'Include' and 'Libs' folders (copied from
echo a compatible Python installation, e.g., 3.10.x) to be present
echo in the SAME directory as this batch script (%~dp0).
echo It will copy them into the .\system\python\ directory.
echo.
echo This script will:
echo 1. Check for and copy 'Include' and 'Libs' folders.
echo 2. Attempt to set up the environment using environment.bat.
echo 3. Verify the existence of the target Python executable (%~dp0system\python\python.exe).
echo 4. Verify and potentially upgrade Pip for the target Python.
echo 5. Install/verify ALL required Python packages using the target Python's Pip:
echo    - torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 (from PyTorch cu126 index)
echo    - xformers (cu126)
echo    - triton-windows==3.2.0.post18
echo    - sageattention (wheel)
echo    - huggingface_hub[cli]
echo    - flash-attention (downloaded wheel)
echo    - pynvml
echo    - jinja2 ^>= 3.1.2
echo    - peft
echo.
echo Press Ctrl+C to cancel, or any other key to continue...
pause > nul
echo.

:: --- Define Paths ---
set "SCRIPT_DIR=%~dp0"
set "TARGET_PYTHON_DIR=%SCRIPT_DIR%system\python\"
set "TARGET_PYTHON_EXE=%TARGET_PYTHON_DIR%python.exe"
set "TARGET_PIP_SCRIPT=%TARGET_PYTHON_DIR%Scripts\pip.exe"
set "ENVIRONMENT_BAT=%SCRIPT_DIR%environment.bat"

:: --- Define Source and Target Paths for Headers/Libs ---
set "SOURCE_INCLUDE_DIR=%SCRIPT_DIR%Include"
set "SOURCE_LIBS_DIR=%SCRIPT_DIR%Libs"
set "TARGET_INCLUDE_DIR=%TARGET_PYTHON_DIR%Include\"
set "TARGET_LIBS_DIR=%TARGET_PYTHON_DIR%Libs\"

:: --- Define Variables for V1 Packages ---
set FLASH_ATTN_REPO_ID=lldacing/flash-attention-windows-wheel
set FLASH_ATTN_FILENAME=flash_attn-2.7.4+cu126torch2.6.0cxx11abiFALSE-cp310-cp310-win_amd64.whl
set "DOWNLOADED_FLASH_ATTN_PATH="
set HF_OUTPUT_TEMP_FILE="%TEMP%\hf_cli_output_%RANDOM%.txt"

:: --- Step 1: Prerequisite Check and Copy Headers/Libs ---
echo [Step 1/7] Checking for and copying Python 'Include' and 'Libs' folders...
echo          Source Dir: %SCRIPT_DIR%
echo          Target Python Dir: %TARGET_PYTHON_DIR%
echo.

:: Check if source Include exists
if not exist "%SOURCE_INCLUDE_DIR%\" (
    echo ERROR: Source folder '%SOURCE_INCLUDE_DIR%' not found.
    echo Please place the 'Include' folder from a suitable Python installation
    echo into the script's directory: %SCRIPT_DIR%
    goto :error_exit
)
echo Found source 'Include' folder.

:: Check if source Libs exists
if not exist "%SOURCE_LIBS_DIR%\" (
    echo ERROR: Source folder '%SOURCE_LIBS_DIR%' not found.
    echo Please place the 'Libs' folder from a suitable Python installation
    echo into the script's directory: %SCRIPT_DIR%
    goto :error_exit
)
echo Found source 'Libs' folder.

:: Ensure target Python directory exists for copying
if not exist "%TARGET_PYTHON_DIR%\" (
    echo INFO: Target Python directory '%TARGET_PYTHON_DIR%' not found.
    echo       Attempting to create directory structure for copy...
    mkdir "%TARGET_PYTHON_DIR%"
    if errorlevel 1 (
        echo ERROR: Failed to create target directory '%TARGET_PYTHON_DIR%'. Check permissions.
        goto :error_exit
    )
    echo       Target directory created.
)

:: Copy Include Folder
echo Copying '%SOURCE_INCLUDE_DIR%' to '%TARGET_INCLUDE_DIR%'...
xcopy "%SOURCE_INCLUDE_DIR%" "%TARGET_INCLUDE_DIR%" /E /H /Y /I /Q
if errorlevel 1 (
    echo ERROR: Failed to copy Include folder. Check permissions and paths.
    echo Command was: xcopy "%SOURCE_INCLUDE_DIR%" "%TARGET_INCLUDE_DIR%" /E /H /Y /I /Q
    goto :error_exit
)
echo 'Include' folder copied successfully.

:: Copy Libs Folder
echo Copying '%SOURCE_LIBS_DIR%' to '%TARGET_LIBS_DIR%'...
xcopy "%SOURCE_LIBS_DIR%" "%TARGET_LIBS_DIR%" /E /H /Y /I /Q
if errorlevel 1 (
    echo ERROR: Failed to copy Libs folder. Check permissions and paths.
    echo Command was: xcopy "%SOURCE_LIBS_DIR%" "%TARGET_LIBS_DIR%" /E /H /Y /I /Q
    goto :error_exit
)
echo 'Libs' folder copied successfully.
echo Python headers and libraries copy complete.
echo.


:: --- Step 2: Environment Setup ---
echo [Step 2/7] Attempting environment setup via environment.bat...
if exist "%ENVIRONMENT_BAT%" (
    call "%ENVIRONMENT_BAT%"
    if errorlevel 1 (
        echo WARNING: Failed to run environment.bat. Environment might be incomplete.
        echo          Attempting to continue using explicitly defined paths...
    ) else (
        echo Environment setup script executed successfully.
    )
) else (
    echo WARNING: environment.bat not found in %SCRIPT_DIR%. Skipping environment setup step.
    echo          Relying solely on explicit paths. Ensure target Python is functional.
)
echo.

:: --- Step 3: Verify Target Python ---
echo [Step 3/7] Verifying target Python executable...
echo           Looking for Python: %TARGET_PYTHON_EXE%
if not exist "%TARGET_PYTHON_EXE%" (
    echo ERROR: Target Python executable not found at '%TARGET_PYTHON_EXE%'.
    echo Please ensure the python executable exists in the 'system\python' subdirectory relative to the script.
    echo The copy step might have created the directory, but python.exe is missing.
    goto :error_exit
)
echo           Found target Python executable.
echo.

:: --- Step 4: Verify Pip for Target Python ---
echo [Step 4/7] Verifying Pip for target Python...
echo           Target Python: %TARGET_PYTHON_EXE%
echo           Checking Pip availability using '%TARGET_PYTHON_EXE% -m pip --version'...
set "PIP_OUTPUT="
for /f "tokens=*" %%a in ('"%TARGET_PYTHON_EXE%" -m pip --version 2^>^&1') do (
    echo   %%a
    set "PIP_OUTPUT=%%a"
)

if not defined PIP_OUTPUT (
    echo WARNING: Failed to get pip version using '%TARGET_PYTHON_EXE% -m pip --version'.
    echo Pip module might not be functioning correctly in '%TARGET_PYTHON_DIR%'.
    echo You may need to run: "%TARGET_PYTHON_EXE%" -m ensurepip --upgrade
    echo Attempting to continue, but pip upgrade and package installs might fail.
    rem Or make this fatal: goto :error_exit
) else (
    echo           Pip verification command executed.
    rem Check if the output string contains the target path AFTER the word "from"
    echo "!PIP_OUTPUT!" | findstr /I /C:" from %TARGET_PYTHON_DIR%" > nul
    if errorlevel 1 (
        rem Secondary check: sometimes path might be slightly different (e.g. Lib vs lib) - less reliable
        echo "!PIP_OUTPUT!" | findstr /I /C:"%TARGET_PYTHON_DIR%" > nul
        if errorlevel 1 (
            echo WARNING: Pip location might be outside '%TARGET_PYTHON_DIR%'. Full output: !PIP_OUTPUT!
            echo          This might indicate user site-packages or PATH issues.
            echo          Using '%TARGET_PYTHON_EXE% -m pip' should mitigate this, continuing...
        ) else (
            echo WARNING: Pip location seems partially related but not exactly '%TARGET_PYTHON_DIR%lib\site-packages\pip'. Full output: !PIP_OUTPUT!
            echo          Using '%TARGET_PYTHON_EXE% -m pip' should mitigate this, continuing...
        )
    ) else (
        echo           Pip location appears consistent with the target Python directory.
    )
)
echo.


:: --- Step 5: Upgrade Pip ---
echo [Step 5/7] Attempting to upgrade Pip for the target Python...
echo           Executing: "%TARGET_PYTHON_EXE%" -m pip install --upgrade pip
"%TARGET_PYTHON_EXE%" -m pip install --upgrade pip
if errorlevel 1 (
    echo WARNING: Failed to upgrade pip. Command exited with error code %errorlevel%.
    echo          Installation of other packages might fail or use an older pip version. Continuing...
) else (
    echo           Pip upgrade command completed. Check output above for success/details.
)
echo.

:: --- Step 6: Define All Packages to Install ---
echo [Step 6/7] Defining list of ALL required packages...
echo           PyTorch Suite: torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 (cu126)
echo           Other V1 Pkgs: xformers, triton-windows, sageattention, huggingface_hub, flash-attention
echo           V2 Packages:   pynvml, jinja2^>=3.1.2, peft
echo.

:: --- Function/Subroutine for Pip Install with Error Check ---
goto :start_installation

:install_package
    echo ------------------------------------------------------------
    echo --- Attempting to install/verify: %*
    echo --- Using command: "%TARGET_PYTHON_EXE%" -m pip install %*
    echo.
    rem --- Directly use %* to pass all arguments received by the subroutine to pip ---
    "%TARGET_PYTHON_EXE%" -m pip install %*
    set INSTALL_ERRORLEVEL=%ERRORLEVEL%
    if %INSTALL_ERRORLEVEL% neq 0 (
        echo ERROR: Failed command: "%TARGET_PYTHON_EXE%" -m pip install %*"
        echo Installation ABORTED. Check pip output and network connection.
        goto :error_exit
    )
    echo.
    echo --- Successfully installed or verified: %*
    echo ------------------------------------------------------------
    echo.
    goto :eof


:start_installation
    :: --- Step 7: Install/Verify ALL Required Packages ---
    echo [Step 7/7] Installing/Verifying ALL required packages using '%TARGET_PYTHON_EXE% -m pip'...
    echo           Target Python: %TARGET_PYTHON_EXE%
    echo.

    :: --- Install PyTorch Suite ---
    echo --- Installing PyTorch Suite (torch, torchvision, torchaudio) ---
    call :install_package torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

	:: --- Packages from V1 (excluding torch/torchvision already handled) ---
	echo --- Installing Other V1 Packages ---
	echo --- WARNING: Installing xformers with --no-deps to prevent forced torch upgrade. ---
	echo ---          This might cause runtime issues if xformers truly requires torch 2.7.0. ---
	call :install_package xformers==0.0.29.post3 --no-deps --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir
	call :install_package triton-windows==3.2.0.post17 --no-cache-dir  
    call :install_package https://github.com/woct0rdho/SageAttention/releases/download/v2.1.1-windows/sageattention-2.1.1+cu126torch2.6.0-cp310-cp310-win_amd64.whl
    call :install_package -U "huggingface_hub[cli]"

    echo --- Downloading flash-attn using huggingface_hub ---
    echo Downloading %FLASH_ATTN_FILENAME% from repo %FLASH_ATTN_REPO_ID% using huggingface-cli...
    set "DOWNLOAD_COMMAND="%TARGET_PYTHON_EXE%" -m huggingface_hub.commands.huggingface_cli download %FLASH_ATTN_REPO_ID% "%FLASH_ATTN_FILENAME%""
    echo Command: %DOWNLOAD_COMMAND%
    echo Redirecting output to %HF_OUTPUT_TEMP_FILE% ...

    :: *** Execute download and redirect output to temp file ***
    %DOWNLOAD_COMMAND% > %HF_OUTPUT_TEMP_FILE% 2>^&1
    set DOWNLOAD_ERRORLEVEL=%ERRORLEVEL%

    :: Read the last line from the temp file
    echo Reading last line from temp file %HF_OUTPUT_TEMP_FILE% ...
    FOR /F "usebackq tokens=*" %%L IN (`type %HF_OUTPUT_TEMP_FILE%`) DO (
        set "DOWNLOADED_FLASH_ATTN_PATH=%%L"
    )

    :: Delete the temporary output file
    if exist %HF_OUTPUT_TEMP_FILE% del %HF_OUTPUT_TEMP_FILE%

    :: Check if the command failed based on the saved errorlevel
    if %DOWNLOAD_ERRORLEVEL% neq 0 (
        echo ERROR: Download command finished with errorlevel %DOWNLOAD_ERRORLEVEL%. Output might be incomplete or contain errors.
        echo Failed command: %DOWNLOAD_COMMAND%
        goto :error_exit
    )

    :: Validate the captured path
    if not defined DOWNLOADED_FLASH_ATTN_PATH (
        echo ERROR: Could not capture download path from output file %HF_OUTPUT_TEMP_FILE%.
        goto :error_exit
    )
    echo Captured potential path: "%DOWNLOADED_FLASH_ATTN_PATH%"
    :: Remove potential surrounding quotes if captured
    set DOWNLOADED_FLASH_ATTN_PATH=%DOWNLOADED_FLASH_ATTN_PATH:"=%
    echo Cleaned path: "%DOWNLOADED_FLASH_ATTN_PATH%"
    if not exist "%DOWNLOADED_FLASH_ATTN_PATH%" (
        echo ERROR: Captured path "%DOWNLOADED_FLASH_ATTN_PATH%" does not exist. Download likely failed or path is incorrect.
        goto :error_exit
    )

    echo Download successful, file located at: "%DOWNLOADED_FLASH_ATTN_PATH%"
    echo.

    echo --- Installing downloaded flash-attn wheel ---
    call :install_package "%DOWNLOADED_FLASH_ATTN_PATH%"

    echo.
    echo --- Installing V2 Packages (excluding torchvision already handled) ---
    call :install_package pynvml
    call :install_package "jinja2>=3.1.2"
    call :install_package peft

goto :success

:success
    echo ======================================================================
    echo == Headers/Libs Copied ^& ALL Required Packages Installed/Verified Successfully! ==
    echo == Target Python: %TARGET_PYTHON_EXE%                               ==
    echo == Packages Checked: torch, torchvision, torchaudio, xformers,       ==
    echo ==                 triton, sageattn, hf_hub, flash-attn, pynvml,     ==
    echo ==                 jinja2, peft                                      ==
    echo ======================================================================
    echo.
    echo Note: 'Requirement already satisfied' means the package was already installed
    echo       and met the version requirements. No action was needed by pip.
    echo Note: Skipping cleanup of downloaded flash-attn file as it resides in the Hugging Face cache.
    echo.
    pause
    exit /b 0

:error_exit
    echo.
    echo ******************************************
    echo **      INSTALLATION FAILED             **
    echo ******************************************
    echo An error occurred during setup or package installation.
    echo Please check the messages above, pip output, and network connectivity.
    :: Cleanup temp file if it somehow still exists on error
    if exist %HF_OUTPUT_TEMP_FILE% del %HF_OUTPUT_TEMP_FILE%
    echo.
    pause
    exit /b 1

:eof
:: End of subroutine marker