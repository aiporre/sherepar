# Möbius centering after CEM

`sherepar` can optionally center a CEM spherical parametrization with a
conformal Möbius transformation. Centering is a post-processing stage: it runs
after CEM's final inverse stereographic projection and does not change the CEM
optimizer or any of its iterations.

This feature does not introduce SEM or another area-preserving optimization.

## Enabling centering

Centering is disabled by default and is available only with CEM:

```bash
python examples/script_to_generate_dataset_from_files.py \
  --input-dir data/source_meshes \
  --output-root data/centered_meshes \
  --param-method cem \
  --mobius-center
```

The deformation dataset generator accepts the same flag:

```bash
python examples/script_to_generate_dataset.py data/meshes \
  --output-root data/generated_centered \
  --param-method cem \
  --mobius-center
```

The Python API uses `mobius_center=True` in
`compute_spherical_parametrization(...)` or
`save_spherical_parametrization(...)`. Combining centering with FLASH raises a
`ValueError`.

## Transform convention and solver

For a center `c` strictly inside the unit ball, the incremental map is

```text
m_c(p) = c + (1 - ||c||²) (p + c) / ||p + c||²,
m_c⁻¹ = m_{-c}.
```

The implementation is in `spherepar/mobius_centering.py`. It uses the original
mesh's normalized Euclidean triangle areas as fixed weights. At every
iteration, the normalized sum of each transformed spherical triangle's three
vertices is its quadrature point. A Poincaré-damped Gauss–Newton solve uses at
most 50 iterations, a centroid tolerance of `1e-10`, and a maximum tangent-step
norm of `2`. Failure to reach the tolerance raises `RuntimeError` and follows
the dataset generator's existing error-log path.

The returned diagnostics include the weighted centroid norm, normalized
spherical area-ratio minimum/maximum/mean, and degenerate/flipped face counts
before and after centering.

## Stored metadata

The spherical OBJ contains the final centered vertices. Its sidecar at
`labels/<sample>_spherical.json` stores the complete ordered sequence under
`metadata.mobius_centering`:

```json
{
  "enabled": true,
  "transform": {
    "kind": "inversion_sequence_v1",
    "centers": [[0.1, -0.02, 0.03], [-0.004, 0.001, 0.002]]
  },
  "iterations": 2,
  "tolerance": 1e-10,
  "max_iterations": 50,
  "max_tangent_step_norm": 2.0,
  "before": {"centroid_norm": 0.12},
  "after": {"centroid_norm": 7.5e-12}
}
```

The abbreviated example omits other diagnostic fields. Forward evaluation
applies centers in stored order. Inverse evaluation applies their negatives in
reverse order. A single final center is not equivalent to this sequence.

Primary labels record the requested setting as
`parametrization.mobius_center`; spherical metadata also records
`metadata.mobius_center`. Resume treats a legacy label with no such field as
uncentered, so an old uncentered sample cannot satisfy a centered request.

## `pmconv` FAUST compatibility

During FAUST preprocessing, `pmconv` reads the transform from the spherical
sidecar referenced by the primary label and saves its centers as
`mobius_centers` in `proc/x_<sample>.npz`. Missing data in an older NPZ is an
identity transform.

`SphereMap` reconstructs the pre-centering CEM sphere once. It then evaluates
the composed map as follows:

```text
forward: mesh query -> CEM interpolation -> complete Möbius transform
inverse: sphere point -> inverse Möbius transform -> inverse CEM lookup
```

The optional center sequence is preserved when a FAUST graph is reconstructed
on another device. With no centers, the original `SphereMap` behavior is
unchanged.
