Get-Location

python bagRunner.py 1 62 bagOutput1.csv &
python bagRunner.py 63 93 bagOutput2.csv &
python bagRunner.py 94 124 bagOutput3.csv &
python bagRunner.py 125 155 bagOutput4.csv &
python bagRunner.py 156 186 bagOutput5.csv &
python bagRunner.py 187 207 bagOutput6.csv &
python bagRunner.py 208 238 bagOutput7.csv &
python bagRunner.py 239 269 bagOutput8.csv &
python bagRunner.py 270 300 bagOutput9.csv &
python bagRunner.py 301 331 bagOutput10.csv &
python bagRunner.py 332 362 bagOutput11.csv &
python bagRunner.py 363 393 bagOutput12.csv &
python bagRunner.py 394 424 bagOutput13.csv &
python bagRunner.py 425 455 bagOutput14.csv &
python bagRunner.py 456 486 bagOutput15.csv &
python bagRunner.py 487 500 bagOutput16.csv &

Get-Job | Wait-Job

Write-Output done