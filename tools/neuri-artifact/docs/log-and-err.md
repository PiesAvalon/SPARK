## Logging

### Modularization

We support the following logging "keys":

- `fuzz`: fuzzing loop;
- `mgen`: model generation;
- `smt`: constraints in smt solving;
- `exec`: model execution;
- `viz`: graphviz visualization;
- `dtest`: dtype_test;
- `core`: seed setting, etc;

The show messages above "INFO" level (see [Python's logging module](https://docs.python.org/3/library/logging.html)). To show debug level message, add `hydra.verbose=[${keys}]` (also see [hydra.logging](https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/logging/)).

```shell
# Show debug information related to `fuzz`:
${NEURI_CMD} hydra.verbose=fuzz
# Show debug info for `fuzz` and `exec`:
${NEURI_CMD} hydra.verbose="[fuzz,exec]"
```

#### Where the log is?

By default, NeuRI logs things both in `console` and `file`. You can find the loggings in [`outputs/${DATE}/${APP}.log`](https://hydra.cc/docs/1.2/tutorials/basic/running_your_app/working_directory/) (current working directory).

## Errors

See `neuri/error.py`:

- `ConstraintError`: Unsatisfiable constraints which is a hint to re-try;
- `InternalError`: NeuRI has some internal bugs that should be fixed.

Takeways:

- Catch `ConstraintError` as a hint to re-try graph generation (no satisfiable solution yet);
- Never catch `InternalError` -- but let the maintainer know the issue and fix it.
