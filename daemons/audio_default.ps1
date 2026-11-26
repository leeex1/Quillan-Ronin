param([switch]$SetToHDMI)

$src = @'
using System;
using System.Runtime.InteropServices;

[Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
public class MMDeviceEnumeratorComObject { }

public enum EDataFlow { eRender = 0, eCapture = 1, eAll = 2 }
public enum ERole { eConsole = 0, eMultimedia = 1, eCommunications = 2 }

[Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IMMDeviceEnumerator {
    int NotImpl1();
    [PreserveSig] int GetDefaultAudioEndpoint(EDataFlow dataFlow, ERole role, out IMMDevice ppDevice);
    int NotImpl2();
    int NotImpl3();
    int NotImpl4();
}

[Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
public interface IMMDevice {
    int Activate(ref Guid iid, int dwClsCtx, IntPtr pActivationParams, [MarshalAs(UnmanagedType.IUnknown)] out object ppInterface);
    int OpenDevicePropertyStore(out object ppProps);
    [PreserveSig] int GetId([MarshalAs(UnmanagedType.LPWStr)] out string ppstrId);
    int GetState(out int pdwState);
}
'@

Add-Type -TypeDefinition $src -ErrorAction SilentlyContinue

$deviceGuid = [Guid]"BCDE0395-E52F-467C-8E3D-C4579291692E"
$enumType = [Type]::GetTypeFromCLSID($deviceGuid)
$enumerator = [System.Activator]::CreateInstance($enumType)
$imm = [IMMDeviceEnumerator]$enumerator

$dev = $null
$hr = $imm.GetDefaultAudioEndpoint([EDataFlow]::eRender, [ERole]::eMultimedia, [ref]$dev)

if ($hr -ne 0 -or $null -eq $dev) {
    Write-Output "DEFAULT_ENDPOINT=QUERY_FAILED hr=$hr"
    exit 1
}

$id = $null
$hr2 = $dev.GetId([ref]$id)
Write-Output "DEFAULT_ENDPOINT_ID=$id"
$state = 0
$dev.GetState([ref]$state) | Out-Null
Write-Output "DEFAULT_ENDPOINT_STATE=$state"

$base = 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Render'
$friendly = 'unknown'
if ($id -and (Test-Path "$base\$id\Properties")) {
    $p = Get-ItemProperty "$base\$id\Properties" -ErrorAction SilentlyContinue
    $friendly = $p.'{b3f8fa53-0004-438e-9003-51a46e139bfc},6'
}
Write-Output "DEFAULT_ENDPOINT_NAME=$friendly"
Write-Output "SetToHDMI=$SetToHDMI"
