# Carta architecture experiments

Carta is the representation-agnostic architecture investigation described in
[`docs/investigation.md`](docs/investigation.md). The code is a CPU/NumPy
reference used to falsify contracts; it is not yet an application framework.

Run the isolated E1 checks from the repository root:

```bash
python -m venv arch/.venv
arch/.venv/bin/pip install -r arch/requirements.txt
PYTHONPATH=arch arch/.venv/bin/python -m unittest discover -s arch/tests -v
```

If compatible NumPy is already installed, the system Python can be used with
the same `PYTHONPATH=arch` command. Carta does not import `nodebased` or modify
the repository-level dependency configuration.
