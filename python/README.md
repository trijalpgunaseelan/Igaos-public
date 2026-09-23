# igaos — Python bindings

Linear, mixed-integer and convex quadratic programming, with no third-party
solver underneath.

```python
from igaos import Model, INF

m = Model(sense="maximize")
x = m.add_variable(0, INF, cost=3.0)
y = m.add_variable(0, INF, cost=5.0)
m.add_constraint({x: 1.0}, upper=4.0)
m.add_constraint({y: 2.0}, upper=12.0)
m.add_constraint({x: 3.0, y: 2.0}, upper=18.0)

r = m.solve()
print(r.objective, r.x)      # 36.0 [2.0, 6.0]
```

The binding is ctypes over the solver's C ABI, so installing it pulls in no
compiler, no numpy and no build step. It needs the shared library
(`libigaos.so`, `libigaos.dylib`, `igaos.dll`), which comes from the C++ build:

```bash
cmake -S .. -B ../build -DCMAKE_BUILD_TYPE=Release
cmake --build ../build -j
export IGAOS_LIBRARY=$PWD/../build/libigaos.so
python -m unittest discover -s tests -v
```

Full reference: `../docs/api.md`.
