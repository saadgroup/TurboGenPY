import os
res = (['32', '64', '128', '256', '512'])
modes = (['100', '1000', '5000', '10000', '20000', '50000', '100000'])
for r in res:
    for m in modes:
        os.system('python example.py -n ' + r + ' -m ' + m + '-gpu')
