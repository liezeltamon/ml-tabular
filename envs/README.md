# Environments

#### `ml-tabular-env`

Environment can be created with:

```bash
bash envs/ml-tabular-env.sh
```

Why not `lazypredict[all]`?

- `lazypredict[all]` pulls in a wider set of optional dependencies
- one of those dependencies tries to install `numba`
- the `numba` version selected during install does not support Python 3.12
- this causes the install to fail with an error saying only Python versions `>=3.8,<3.12` are supported

`lazypredict[boost]` worked on Python 3.12 and is a better fit here if you mainly want boosted tree models in the benchmark shortlist.
