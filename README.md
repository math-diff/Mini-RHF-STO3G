## Usage

Basic run:
````bash
python RHF_(h_f_atom).py --pdb your_structure.pdb
````

## Limitations

- Supports only elements H–F (Z = 1–9). To include more elements you must extend sto3g_data. Integrals involving d (and higher) functions are not tested.
- Implements only closed–shell RHF.
- No special numerical stabilization or high‑accuracy enhancements (the simple Boys function recursion may accumulate error for high angular momentum or large inter-center distances).
- Intended for educational / experimental purposes; not for production or high‑precision research.
