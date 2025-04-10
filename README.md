# Bayesian Inversion with Probabilistic Surrogates

TODO:
- Discuss runner vs. analysis scripts

## Simulation study workflow

### Inverse problem and algorithm setup
These scripts are only run once at the beginning of each experiment.
1. `run_inv_prob_setup.r`
2. `run_save_alg_settings.r`

### Round 1: Initial Design
The first round is special in that the emulators are created and fit to 
initial designs; in subsequent rounds, these designs are sequentially 
augmented and the emulators are updated. The structure is set up so that 
replicates are created for each initial design type, so that subsequent 
algorithms can be analyzed on on average sense over the randomness in the
initial design.

1. `run_init_design.r`
2. `run_init_emulator.r`
At present, this saves outputs (fit emulators) to 
`experiments/<experiment_tag>/output/round1/em/<em_tag>/<em_id>`. Recall that 
emulator IDs are unique within each round and emulator tag. This output 
directory is somewhat inconsistent with the code at later stages. To make it 
consistent, it should be saving to `.../em/<em_tag>/<design_tag>/<design_id>/<em_id>`.


# Understanding the ID/tag structure

em_tag, mcmc_tag, and acq_tag are constant throughout an experiment (they
uniquely define an emulator model, MCMC algorithm, and sequential design/acquisition 
method, respectively). Fixed sets of these tags are defined at the beginning 
of the experiment. 

Let y = f_i(x) mean that y is uniquely defined as a function of x, conditional 
on being in round i. Here "x" is a dependency from within round "i". Also let 
y = f_i(x|z) mean that y is uniquely defined as a function of x and z, where
the z dependency comes from the previous round, i-1. Dependencies will always
have this Markov property. Note that all of these IDs are conditional on 
a specific experiment tag, which is not made explicit in the notation
for the sake of brevity.

## Round 1
Round one is unique in that it does not use the acq_tags at all, as these 
specifically handle sequential, not initial design. The design tags used
in this round are different, and are created on the fly. An example is 
"LHS_200", a tag indicating a latin hypercube sample with 200 initial design
points. The design (acquisition) and emulator steps in round one differ from 
the analogous steps in future rounds. However, the MCMC step is the same 
for all rounds. To avoid confusion we use "design_tag" to refer to the initial
design method in round 1, and "acq_tag" to refer to the acquisition/sequential
design methods in subsequent rounds.

design_id = f_1(design_tag)
em_id = f_1(em_tag, design_tag, design_id) // Initial emulator depends on the emulator model and the initial design used to fit the model.
mcmc_id = f_1(mcmc_tag, em_tag, em_id)     // MCMC run depends on the MCMC algorithm, and a specific fit emulator model.

## Round i, i >= 2
All other rounds correspond to the sequential design setting. No new emulators 
are fit; instead, existing emulators are updated by adding additional 
design points. Thus, these rounds depend on the outputs of the immediately 
preceding round.

acq_id = f_i(acq_tag | mcmc_tag, mcmc_id) // New design depends on the acquisition method and a specific MCMC run from the previous round (which also maps to a specific emulator model).
em_id = f_i(em_tag, acq_tag, acq_id)
mcmc_id = f_i(mcmc_tag, em_tag, em_id)

# Output directory structure
The output directory structure tries to capture the dependency structure
described above.

## Round 1
The below trees summarize the directory structure starting at the level 
`experiments/<experiment_tag>/output/`.

```
round1/
├── design/
|   ├── id_map.csv
│   ├── <design_tag>/
│   |   |   └── design_<design_id>/
│   |   |   |   └── design_info.rds
├── em/
|   ├── id_map.csv
│   ├── <em_tag>/
│   |   ├── <design_tag>/
│   |   |   ├── design_<design_id>/
│   |   |   |   └── em_<em_id>/
│   |   |   |   |   └── em_llik.rds
├── mcmc/
|   ├── id_map.csv
│   ├── <mcmc_tag>/
|   |   ├── <em_tag>/
|   |   |   ├── em_<em_id>/
|   |   |  |   ├── mcmc_<mcmc_id>/
|   |   |  |   |   └── mcmc_samp.rds
```

TODO: at what level should the id maps be defined? seems like this will be
based on how the jobs are batched out. e.g., if a script handles a specific
(mcmc_tag, em_tag) combo, then it makes sense to write it to mcmc_tag/em_tag.
Actually this is not necessary: if we assume the runs always are initiated 
from a central place (the runner files), then the runner files can be in
charge of creating the id maps for all runs. So maybe it makes sense to put
the id maps at the top levels.

## All other rounds
We use round 2 as an example.

```
round2/
├── design/
|   ├── id_map.csv
│   ├── <acq_tag>/
│   |   ├── <mcmc_tag>/
│   |   |   ├── mcmc_<mcmc_id>/
│   |   |   |   |── design_<acq_id>/
│   |   |   |   |   └── design_info.rds
├── em/
|   ├── id_map.csv
│   ├── <em_tag>/
│   |   ├── <design_tag>/
│   |   |   ├── design_<design_id>/
│   |   |   |   └── em_<em_id>/
│   |   |   |   |   └── em_llik.rds
├── mcmc/
|   ├── id_map.csv
│   ├── <mcmc_tag>/
|   |   ├── <em_tag>/
|   |   |   ├── em_<em_id>/
|   |   |  |   ├── mcmc_<mcmc_id>/
|   |   |  |   |   └── mcmc_samp.rds
```










