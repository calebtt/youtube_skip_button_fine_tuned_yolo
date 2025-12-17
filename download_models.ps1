# Download YOLOv11n (works, as you confirmed)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt" -OutFile "yolo11n.pt"

# Retry SAM-2 with robust flags (YOLO already done)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2_b.pt" -OutFile "sam2_b.pt" -UseBasicParsing -TimeoutSec 300

# Verify both files (sizes should match)
dir *.pt | Select-Object Name, @{Name="Size (MB)"; Expression={[math]::Round($_.Length / 1MB, 2)}}