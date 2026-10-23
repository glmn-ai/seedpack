import sys, re; sys.path.insert(0, 'src')
from gpuquant import GPUQuant
try:
    g = GPUQuant(); print('ok')
except Exception as e:
    s = str(e); errs = [l for l in s.split('\n') if 'error' in l]
    print(len(errs), 'errors')
    lines = s.split('\n')
    for i, l in enumerate(lines):
        if ': error' in l:
            print(re.sub(r'.*cubin\.cu', '', l)); print('   ', lines[i + 1].strip()[:150])
    if not errs: print(s[-3000:])
