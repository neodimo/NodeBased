Unicode true
!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"
Name "NodeBased"
OutFile "..\release\NodeBased-${VERSION}-windows-x64-setup.exe"
InstallDir "$LOCALAPPDATA\Programs\NodeBased"
InstallDirRegKey HKCU "Software\NodeBased" "InstallDir"
RequestExecutionLevel user
SetCompressor /SOLID lzma
!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\NodeBased.exe"
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "English"
Var Params
Var Restart
Var WaitPid
Var Process

Function .onInit
  ${GetParameters} $Params
  StrCpy $Restart ""
  ClearErrors
  ${GetOptions} $Params "/RESTART" $Restart
  ${IfNot} ${Errors}
    StrCpy $Restart "yes"
  ${EndIf}
  ${GetOptions} $Params "/WAITPID=" $WaitPid
  ${If} $WaitPid != ""
    System::Call 'kernel32::OpenProcess(i 0x100000, i 0, i $WaitPid) p.r0'
    StrCpy $Process $0
    ${If} $Process != 0
      System::Call 'kernel32::WaitForSingleObject(p $Process, i 60000) i.r0'
      System::Call 'kernel32::CloseHandle(p $Process)'
      ${If} $0 != 0
        MessageBox MB_OK "NodeBased is still closing. Please close it and run this installer again."
        Abort
      ${EndIf}
    ${EndIf}
  ${EndIf}
FunctionEnd

Section "NodeBased"
  SetOutPath "$INSTDIR"
  File /r "..\dist\NodeBased\*"
  WriteRegStr HKCU "Software\NodeBased" "InstallDir" "$INSTDIR"
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  CreateDirectory "$SMPROGRAMS\NodeBased"
  CreateShortcut "$SMPROGRAMS\NodeBased\NodeBased.lnk" "$INSTDIR\NodeBased.exe"
  CreateShortcut "$SMPROGRAMS\NodeBased\Uninstall.lnk" "$INSTDIR\Uninstall.exe"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NodeBased" "DisplayName" "NodeBased"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NodeBased" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NodeBased" "UninstallString" '$"$INSTDIR\Uninstall.exe$"'
  WriteRegStr HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NodeBased" "InstallLocation" "$INSTDIR"
SectionEnd

Function .onInstSuccess
  ${If} $Restart == "yes"
    Exec '"$INSTDIR\NodeBased.exe"'
  ${EndIf}
FunctionEnd

Section "Uninstall"
  Delete "$INSTDIR\NodeBased.exe"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir /r "$INSTDIR\_internal"
  RMDir "$INSTDIR"
  Delete "$SMPROGRAMS\NodeBased\NodeBased.lnk"
  Delete "$SMPROGRAMS\NodeBased\Uninstall.lnk"
  RMDir "$SMPROGRAMS\NodeBased"
  DeleteRegKey HKCU "Software\NodeBased"
  DeleteRegKey HKCU "Software\Microsoft\Windows\CurrentVersion\Uninstall\NodeBased"
SectionEnd
