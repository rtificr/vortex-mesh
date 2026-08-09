# VortexMesh

Converts OBJ files into Vortex projects. It can both create a new project and write to an existing one.

## Install

You can either download a prebuilt binary from the [releases page](https://github.com/rtificr/vortex-mesh/releases) or build from source with Cargo:
```bash
cargo install vortex-mesh
```

## Usage

```bash
vortex-mesh model.obj
```

The command writes `model.json` beside the source OBJ by default.

```bash
vortex-mesh model.obj \
  --output scene.json \
  --position 10 0 -5 \
  --scale 2 \
  --color FF8800 \
  --material Wood
```

Use a single scale value for uniform scaling, or three values for per-axis
scaling:

```bash
vortex-mesh model.obj --scale 2 1 0.5
```

To add generated parts to an existing project, provide `--project`. Unless
`--output` is also supplied, that project file is updated in place.

```bash
vortex-mesh model.obj --project existing-project.json
```

## Important options

| Option | Description |
| --- | --- |
| `--position X Y Z` | Translates the imported mesh. |
| `--scale X [Y] [Z]` | Applies positive uniform or per-axis scaling. |
| `--color RRGGBB[AA]` | Sets the generated part color in hexadecimal. |
| `--material NAME` | Sets the generated part material. Defaults to `Plastic`. |
| `--max-relative-surface-error VALUE` | Controls approximation quality. Lower values create more parts. Defaults to `0.01`. |
| `--project FILE` | Loads a Vortex project and appends the generated parts. |
| `--output FILE` | Selects the JSON output file. |

Run `vortex-mesh --help` for the complete command reference.

## Development

```bash
cargo test
cargo clippy -- -D warnings
```

## License

MIT
