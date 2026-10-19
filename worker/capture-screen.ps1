Add-Type -Assembly System.Windows.Forms
Add-Type -Assembly System.Drawing

$scale = 0.5
$w = 960
$h = 540

try {
    $s = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
    $w = [int]($s.Width * $scale)
    $h = [int]($s.Height * $scale)
    $src = New-Object System.Drawing.Bitmap($s.Width, $s.Height)
    $g2 = [System.Drawing.Graphics]::FromImage($src)
    $g2.CopyFromScreen(0, 0, 0, 0, $src.Size)
    $g2.Dispose()

    $b = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.DrawImage($src, 0, 0, $w, $h)
    $g.Dispose()
    $src.Dispose()
} catch {
    # Fallback to a solid neutral canvas if screen capture is unavailable in current session
    $b = New-Object System.Drawing.Bitmap($w, $h)
    $g = [System.Drawing.Graphics]::FromImage($b)
    $g.Clear([System.Drawing.Color]::FromArgb(30, 30, 30))
    $g.Dispose()
}

$ms = New-Object System.IO.MemoryStream
$codec = [System.Drawing.Imaging.ImageCodecInfo]::GetImageEncoders() | Where-Object { $_.MimeType -eq 'image/jpeg' }
$ep = New-Object System.Drawing.Imaging.EncoderParameters(1)
$ep.Param[0] = New-Object System.Drawing.Imaging.EncoderParameter([System.Drawing.Imaging.Encoder]::Quality, 70L)
$b.Save($ms, $codec, $ep)
$bytes = $ms.ToArray()
$ms.Dispose()
$b.Dispose()
[Convert]::ToBase64String($bytes)
