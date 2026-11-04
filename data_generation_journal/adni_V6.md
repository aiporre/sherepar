# ADNI and FAUST spherical parametrization notes

This journal records the SphereMap/CMCF runs, comparisons with CEM, mesh
smoothing experiments, and the remaining orientation issue for the ADNI data.

## FAUST

Install the MoebiusRegistration Python package in the active environment before
running the SphereMap backend:

```bash
python -m pip install /home/sauron/Documents/Phd/code/MoebiusRegistration
```

The following SphereMap (CMCF) settings gave the best results in this run. The
maps had fewer collapsed triangles than the CEM result. About 35% of vertices
were reported as collapsed/at actual zero; this remained a limitation of the
result.

```bash
python examples/script_to_generate_dataset_from_files.py \
  --input-dir /media/sauron/GG2/datasets/deformed_blobs/faust_6890/templates/registrations/ \
  --output-root /media/sauron/GG2/datasets/deformed_blobs/faust_6890/data_CMCF \
  --param-method spheremap \
  --spheremap-repository /home/sauron/Documents/Phd/code/MoebiusRegistration \
  --spheremap-iters 25 \
  --spheremap-step-size 1.0 \
  --spheremap-threads 4 \
  --spheremap-degree 1 \
  --spheremap-a-steps 10 \
  --spheremap-a-step-size 0.05 \
  --spheremap-poincare-max-norm 2.0 \
  --spheremap-c2i 0 \
  --spheremap-gss-tolerance 1e-6 \
  --spheremap-verbose \
  --no-resume
```

The CEM comparison showed a completely flipped sphere for `tr_reg_003`:

![[tr_reg_003_mesh.png]]

![[tr_reg_003_sphere.png]]

The corresponding CMCF plots showed lower errors:

![[tr_reg_003_mesh 1.png]]

![[tr_reg_003_sphere 1.png]]

Smoothing the FAUST meshes improved the results only marginally. The quality
smoother command used was:

```bash
python examples/smooth_registration_dataset.py \
  --input-dir /media/sauron/GG2/datasets/deformed_blobs/faust_6890/registrations \
  --output-dir /media/sauron/GG2/datasets/deformed_blobs/faust_6890/registrations_smoothed \
  --pattern '*.ply' \
  --iterations 20 \
  --angle-threshold 15
```

Related reports:

1. [[report error sphere par on FAUST with laplacian]]
2. [[report error sphere par on FAUST with naive smooth shifts]]

Plot a dataset-wide error report with:

```bash
python examples/plot_dataset_spherical_errors.py \
  --root /media/sauron/GG2/datasets/deformed_blobs/faust_6890/data_CMCF_mobius \
  --backend cmcf \
  --output-root data/faust_errors_CMCF_mobius_center
```

The plotter's option is `--output-root` (not `--output`).

## ADNI

### CEM inspection

CEM gave good results for most ADNI meshes, with exceptions around very thin
triangles. The following plots document the thin triangles and collapsed faces:

![[ADNI1_cem_angles_below_30deg.png]]

![[ADNI1_cem_negative_cotangent_weights.png]]

![[ADNI1_cem_original_mesh.png]]

![[ADNI1_cem.png]]

![[ADNI1_cem_anchor_distance.png]]

These observations motivated using CMCF/SphereMap for the next run.

### Initial SphereMap run

The run selected the first session and left hip for each participant. The
reported dataset contained 1,833 subjects in MNI space. The filters were
asymmetric in the original data, so the run retained only the left hip and the
first session matching the screening diagnosis in `diagnosis_sc`, avoiding
diagnosis conversion across visits.

```bash
MPLCONFIGDIR=/tmp/sherepar-mpl python examples/script_to_generate_dataset_from_files.py \
  --input-dir /media/sauron/GG2/datasets/ADNI/fixmodels_mni_17k/fixmodels_mni \
  --adni-participants /media/sauron/GG2/datasets/ADNI/participants.tsv \
  --adni-session first \
  --adni-hip left \
  --param-method spheremap \
  --spheremap-binary /home/sauron/Documents/Phd/code/MoebiusRegistration/Bin/Linux/SphereMap \
  --spheremap-iters 25 \
  --spheremap-step-size 1.0 \
  --spheremap-threads 4 \
  --output-root /media/sauron/GG2/datasets/deformed_blobs/adni_3000/data_cmcf \
  --no-resume
```

Plot the output using the dataset-wide spherical error plotter:

```bash
MPLCONFIGDIR=/tmp/sherepar-mpl python examples/plot_dataset_spherical_errors.py \
  --root /media/sauron/GG2/datasets/deformed_blobs/adni_3000/data_cmcf \
  --backend cmcf \
  --output-root /media/sauron/GG2/datasets/deformed_blobs/adni_3000/adni_error_report_cmcf
```

The first plots showed substantial errors:

![[sub-ADNI002S4219-ses-M000_hip_left_mesh.png]]

![[sub-ADNI002S4219-ses-M000_hip_left_sphere.png]]

### Smooth templates and rerun SphereMap

The initial result motivated smoothing the templates and rerunning
parametrization. This Laplacian smoothing command uses a step of 0.2 and a
two-ring neighborhood, and saves a plot for each input mesh:

```bash
python examples/smooth_registration_dataset.py \
  --input-dir /media/sauron/GG2/datasets/deformed_blobs/adni_3000/templates_filtered \
  --output-dir /media/sauron/GG2/datasets/deformed_blobs/adni_3000/templates_filtered_smooth \
  --pattern '*.obj' \
  --method laplacian \
  --laplacian-step 0.2 \
  --laplacian-rings 2 \
  --plot
```

![[sub-ADNI002S0295-ses-M000_hip_left.png]]

Run SphereMap on the smoothed meshes:

```bash
MPLCONFIGDIR=/tmp/sherepar-mpl python examples/script_to_generate_dataset_from_files.py \
  --input-dir /media/sauron/GG2/datasets/deformed_blobs/adni_3000/templates_filtered_smooth \
  --adni-participants /media/sauron/GG2/datasets/ADNI/participants.tsv \
  --adni-session first \
  --adni-hip left \
  --param-method spheremap \
  --spheremap-binary /home/sauron/Documents/Phd/code/MoebiusRegistration/Bin/Linux/SphereMap \
  --spheremap-iters 25 \
  --spheremap-step-size 1.0 \
  --spheremap-threads 4 \
  --output-root /media/sauron/GG2/datasets/deformed_blobs/adni_3000/data_cmcf_smoot \
  --no-resume
```

### Remaining orientation issue

The saved maps are often completely folded inward. The CEM importer has a
`--force-outward-winding` option, but that option is restricted to
`--param-method cem`; the current CLI rejects it for SphereMap. It also
reverses face winding globally and can hide local fold diagnostics. Do not add
it to the SphereMap commands above. How downstream PMConv should handle the
inward-oriented SphereMap faces remains unresolved.

Example plots from the smoothed SphereMap run:

![[sub-ADNI002S0295-ses-M000_hip_left_mesh.png]]

![[sub-ADNI002S0295-ses-M000_hip_left_sphere.png]]
