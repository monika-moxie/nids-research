# Paper Figures

Generate the tracked paper figures from saved experiment outputs:

```powershell
.\.venv\Scripts\python.exe paper\generate_figures.py
```

The script produces four PNG files from the saved JSON metrics. Do not edit
their values manually; rerun the script after a validated experiment changes.

The bridge plot intentionally shows clean F1 beside certified accuracy at
radius 0.10. Its purpose is to prevent a certificate value from being read as
an overall quality score when clean utility differs substantially.
