# Run verification: measurement-synthetic

- synthetic: True
- integrity_pass: True
- complete: True (120/120 pairs)
- errors: 0, contaminated: 0, pair_incomplete: 0

| case | mode | valid/expected | n | attempts | SF/FS | errors | contaminated |
|---|---|---|---|---|---|---|---|
| P1-n65536-f32-contig-t | direct | 30/30 | 2081 | 1 | 15/15 | 0 | 0 |
| P1-n65536-f32-contig-t | replay | 30/30 | 2905 | 1 | 15/15 | 0 | 0 |
| R1-m256-k512-f32-contig-t | direct | 30/30 | 1352 | 1 | 15/15 | 0 | 0 |
| R1-m256-k512-f32-contig-t | replay | 30/30 | 1657 | 1 | 15/15 | 0 | 0 |
