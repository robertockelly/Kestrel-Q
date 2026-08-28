[CmdletBinding()]
param(
    [string]$MachineId = 'KQ-01',
    [string]$CaptureDate = '2026-08-28',
    [string]$BenchmarkExe,
    [int]$SampleIntervalMilliseconds = 100
)

$ErrorActionPreference = 'Stop'

if ($SampleIntervalMilliseconds -lt 50) {
    throw 'SampleIntervalMilliseconds must be at least 50.'
}

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$repositoryRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $scriptDirectory '..'))

if ([string]::IsNullOrWhiteSpace($BenchmarkExe)) {
    $BenchmarkExe = Join-Path $repositoryRoot (
        'build-cuda\Release\kq_cuda_bandwidth.exe')
}
$BenchmarkExe = (Resolve-Path -LiteralPath $BenchmarkExe).Path

function Resolve-NvidiaSmi {
    $command = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    $systemCandidate = Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'
    if (Test-Path -LiteralPath $systemCandidate) {
        return $systemCandidate
    }

    $programFilesCandidate = Join-Path $env:ProgramFiles (
        'NVIDIA Corporation\NVSMI\nvidia-smi.exe')
    if (Test-Path -LiteralPath $programFilesCandidate) {
        return $programFilesCandidate
    }

    throw 'nvidia-smi.exe could not be resolved.'
}

$nvidiaSmi = Resolve-NvidiaSmi
$evidenceDirectory = Join-Path $repositoryRoot (
    "bench\results\raw\$MachineId\$CaptureDate")
$csvPath = Join-Path $evidenceDirectory 'cuda-bandwidth.csv'
$consolePath = Join-Path $evidenceDirectory 'cuda-bandwidth-console.txt'
$samplesPath = Join-Path $evidenceDirectory 'pcie-link-samples.csv'

foreach ($path in @($csvPath, $consolePath, $samplesPath)) {
    if (Test-Path -LiteralPath $path) {
        throw "Refusing to overwrite immutable raw evidence: $path"
    }
}

$temporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) (
    'kq-cuda-bandwidth-' + [System.Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $temporaryDirectory | Out-Null

$temporaryCsv = Join-Path $temporaryDirectory 'cuda-bandwidth.csv'
$temporaryError = Join-Path $temporaryDirectory 'benchmark-stderr.txt'
$temporaryConsole = Join-Path $temporaryDirectory 'cuda-bandwidth-console.txt'
$temporarySamples = Join-Path $temporaryDirectory 'pcie-link-samples.csv'

$metadata = [System.Collections.Generic.List[string]]::new()
$metadata.Add('Kestrel-Q Task 0.2E raw capture')
$metadata.Add("capture_started_utc=$([DateTime]::UtcNow.ToString('o'))")
$metadata.Add("machine_id=$MachineId")
$metadata.Add("capture_date=$CaptureDate")
$metadata.Add("benchmark_executable=$BenchmarkExe")
$metadata.Add("nvidia_smi=$nvidiaSmi")
$metadata.Add("sample_interval_ms=$SampleIntervalMilliseconds")
$metadata.Add("git_commit=$(git -C $repositoryRoot rev-parse HEAD)")
$metadata.Add('')
$metadata.Add('cmake --version:')
$metadata.AddRange([string[]](& cmake --version 2>&1))
$metadata.Add('')
$metadata.Add('nvcc --version:')
$metadata.AddRange([string[]](& nvcc --version 2>&1))
$metadata.Add('')
$metadata.Add('nvidia-smi:')
$metadata.AddRange([string[]](& $nvidiaSmi 2>&1))
$metadata | Set-Content -LiteralPath $temporaryConsole -Encoding utf8

$arguments = @(
    '--machine-id', $MachineId,
    '--warmup', '3',
    '--iterations', '10'
)

$samples = [System.Collections.Generic.List[object]]::new()
$process = Start-Process `
    -FilePath $BenchmarkExe `
    -ArgumentList $arguments `
    -RedirectStandardOutput $temporaryCsv `
    -RedirectStandardError $temporaryError `
    -WindowStyle Hidden `
    -PassThru

try {
    while (-not $process.HasExited) {
        $sampleTimestamp = [DateTime]::UtcNow.ToString('o')
        $sampleOutput = & $nvidiaSmi `
            '--query-gpu=timestamp,name,pstate,pcie.link.gen.current,pcie.link.width.current,memory.total,memory.used,memory.free' `
            '--format=csv,noheader,nounits' 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "nvidia-smi link sampling failed: $sampleOutput"
        }

        $fields = @($sampleOutput -split ',' | ForEach-Object { $_.Trim() })
        if ($fields.Count -ne 8) {
            throw "Unexpected nvidia-smi sample format: $sampleOutput"
        }
        $samples.Add([pscustomobject]@{
            sample_timestamp_utc = $sampleTimestamp
            nvidia_timestamp = $fields[0]
            gpu_name = $fields[1]
            pstate = $fields[2]
            pcie_link_gen_current = $fields[3]
            pcie_link_width_current = $fields[4]
            memory_total_mib = $fields[5]
            memory_used_mib = $fields[6]
            memory_free_mib = $fields[7]
        })
        Start-Sleep -Milliseconds $SampleIntervalMilliseconds
        $process.Refresh()
    }
    $process.WaitForExit()

    Add-Content -LiteralPath $temporaryConsole -Encoding utf8 -Value @(
        ''
        "benchmark_exit_code=$($process.ExitCode)"
        "capture_finished_utc=$([DateTime]::UtcNow.ToString('o'))"
        "pcie_sample_count=$($samples.Count)"
        ''
        'benchmark stderr:'
    )
    if (Test-Path -LiteralPath $temporaryError) {
        Get-Content -LiteralPath $temporaryError |
            Add-Content -LiteralPath $temporaryConsole -Encoding utf8
    }

    if ($process.ExitCode -ne 0) {
        throw "Bandwidth benchmark failed with exit code $($process.ExitCode)."
    }
    if ($samples.Count -eq 0) {
        throw 'No PCIe link samples were captured while the benchmark ran.'
    }

    $samples | Export-Csv -LiteralPath $temporarySamples `
        -NoTypeInformation -Encoding utf8

    New-Item -ItemType Directory -Path $evidenceDirectory -Force | Out-Null
    Move-Item -LiteralPath $temporaryCsv -Destination $csvPath
    Move-Item -LiteralPath $temporaryConsole -Destination $consolePath
    Move-Item -LiteralPath $temporarySamples -Destination $samplesPath

    Write-Output "Captured immutable raw evidence in $evidenceDirectory"
} catch {
    Write-Error (
        "Capture failed. Preserved temporary evidence at ${temporaryDirectory}: " +
        $_.Exception.Message)
    exit 1
}
