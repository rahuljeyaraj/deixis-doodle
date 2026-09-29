# Synthesise test utterances with Windows' built-in System.Speech (SAPI) voices and record
# each word's start time from the SpeakProgress events.
#
# Called by scripts/make_test_utterances.py; you normally don't run it directly.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/make_test_utterances.ps1 `
#       -Jobs jobs.json -Out events.json
#
# jobs.json: [{"id": "...", "text": "...", "voice": "Microsoft Zira Desktop", "rate": 0,
#              "wav": "C:\\...\\x.wav"}, ...]
# events.json: [{"id": "...", "wav": "...", "words": [{"text", "char", "ms"}], "total_ms"}]
# Output WAVs are 16 kHz, 16-bit, mono PCM.

param(
    [Parameter(Mandatory = $true)][string]$Jobs,
    [Parameter(Mandatory = $true)][string]$Out
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

# SpeakProgress fires on the synthesiser's thread; a small C# collector is more reliable than
# PowerShell's asynchronous event queue.
Add-Type -ReferencedAssemblies System.Speech -TypeDefinition @"
using System;
using System.Collections.Generic;
using System.Speech.Synthesis;

public class WordCollector {
    public List<object[]> Words = new List<object[]>();
    public void Attach(SpeechSynthesizer s) {
        s.SpeakProgress += (sender, e) => {
            Words.Add(new object[] { e.Text, e.CharacterPosition, e.AudioPosition.TotalMilliseconds });
        };
    }
}
"@

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(
    16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)

$results = @()
foreach ($job in (Get-Content -Raw -Encoding UTF8 $Jobs | ConvertFrom-Json)) {
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $synth.SelectVoice($job.voice)
    $synth.Rate = [int]$job.rate
    $collector = New-Object WordCollector
    $collector.Attach($synth)
    $synth.SetOutputToWaveFile($job.wav, $fmt)
    $synth.Speak([string]$job.text)
    $synth.SetOutputToNull()
    $synth.Dispose()

    $words = @()
    foreach ($w in $collector.Words) {
        $words += [ordered]@{ text = $w[0]; char = $w[1]; ms = $w[2] }
    }
    $results += [ordered]@{ id = $job.id; wav = $job.wav; words = $words }
}

ConvertTo-Json -InputObject $results -Depth 5 | Out-File -Encoding utf8 $Out
Write-Output "wrote $($results.Count) utterances"
